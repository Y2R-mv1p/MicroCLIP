"""Standalone training; evaluation is provided by evaluate.py at the repository root."""
from open_clip_train.train_eval_only_sevenfive import (
    ClipLoss,
    CsvImageLabelDataset,
    CsvImageTextDataset,
    DDP,
    DataLoader,
    DistributedSampler,
    SemanticCaptionLookup,
    TRAIN_ONLY_METRICS_FIELDS,
    _LOGGER,
    append_csv,
    build_wds_train_loader,
    cleanup_distributed,
    create_optimizer,
    create_scheduler,
    decode_separator,
    dist,
    install_error_logging,
    is_main_process,
    load_resume,
    looks_like_wds,
    maybe_progress,
    open_clip,
    os,
    parse_args,
    read_class_names,
    resolve_output_dir,
    save_checkpoint,
    save_epoch_weights,
    save_json,
    set_seed,
    setup_distributed,
    setup_logging,
    time,
    torch,
    train_one_epoch,
    train_one_epoch_text_prototype,
    validate_args,
)


def main():
    args = parse_args(training_only=True)
    validate_args(args)
    args.csv_separator = decode_separator(args.csv_separator)
    setup_distributed(args)
    set_seed(args.seed, args.rank)

    output_dir = resolve_output_dir(args)
    if not args.resume and output_dir.exists() and any(output_dir.glob("checkpoints/**/*.pt")):
        raise FileExistsError(f"Existing checkpoints in {output_dir}; use --resume or a new --output-dir.")
    checkpoint_dir = output_dir / "checkpoints"
    if is_main_process(args):
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.distributed:
        dist.barrier()

    logger = setup_logging(output_dir, args)
    install_error_logging(output_dir, args, logger)
    progress = is_main_process(args) and not args.no_progress
    if progress and maybe_progress is None:
        logger.warning("tqdm is not installed; progress bars are disabled.")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    logger.info(
        "Starting training-only run | objective=%s rank=%d world_size=%d device=%s output_dir=%s",
        args.train_objective,
        args.rank,
        args.world_size,
        args.device,
        output_dir,
    )
    logger.info("Train data: %s", args.train_data)

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        force_quick_gelu=args.force_quick_gelu,
        force_image_size=args.force_image_size,
        force_context_length=args.force_context_length,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        cache_dir=args.cache_dir,
    )
    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    tokenizer = open_clip.get_tokenizer(args.model, cache_dir=args.cache_dir, context_length=args.force_context_length)
    use_wds = looks_like_wds(args.train_data)
    if use_wds and args.train_objective != "contrastive":
        raise ValueError("WebDataset training currently supports --train-objective contrastive only.")

    semantic_lookup = None
    if args.semantic_structure_tsv:
        semantic_lookup = SemanticCaptionLookup(args.semantic_structure_tsv, args.semantic_cell_tsv)
        if not use_wds:
            raise ValueError(
                "Semantic title replacement requires WebDataset training data "
                "(canonical_organelle/canonical_cell come from the sample json)."
            )
        if is_main_process(args):
            _LOGGER.info(
                "Semantic captions loaded | structures=%d cells=%d replace_mode=%s",
                len(semantic_lookup.structure_captions),
                len(semantic_lookup.cell_captions),
                args.semantic_replace_mode,
            )

    train_classes = []
    train_class_to_idx = {}
    train_class_tokens = None
    if not use_wds:
        train_classes = read_class_names(args.train_data, args.csv_caption_key, args.csv_separator)
        train_class_to_idx = {name: idx for idx, name in enumerate(train_classes)}
        train_class_tokens = tokenizer([args.template.format(name=name) for name in train_classes])

    if is_main_process(args):
        save_json(output_dir / "args.json", vars(args))
        save_json(
            output_dir / "class_summary.json",
            {
                "train_classes": len(train_classes),
            },
        )
        if train_class_to_idx:
            save_json(output_dir / "train_class_to_idx.json", train_class_to_idx)

    train_shared_epoch = None
    if use_wds:
        train_loader, train_shared_epoch, train_sample_count = build_wds_train_loader(
            args,
            preprocess_train,
            tokenizer,
            semantic_lookup=semantic_lookup,
        )
        train_dataset = None
        train_sampler = None
    elif args.train_objective == "text-prototype":
        train_dataset = CsvImageLabelDataset(
            args.train_data,
            args.csv_img_key,
            args.csv_caption_key,
            preprocess_train,
            train_class_to_idx,
            sep=args.csv_separator,
        )
    else:
        train_dataset = CsvImageTextDataset(
            args.train_data,
            args.csv_img_key,
            args.csv_caption_key,
            preprocess_train,
            tokenizer,
            sep=args.csv_separator,
            long_caption_key=args.train_long_caption_key,
            long_caption_mode=args.train_long_caption_mode,
            long_caption_chunk_tokens=args.long_caption_chunk_tokens,
            long_caption_chunk_overlap=args.long_caption_chunk_overlap,
            long_caption_prepend_title=args.long_caption_prepend_title,
            long_caption_sample_prob=args.long_caption_sample_prob,
        )
    if not use_wds:
        train_sampler = (
            DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
            if args.distributed
            else None
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=device.type == "cuda" and not args.disable_pin_memory,
            persistent_workers=args.persistent_workers and args.workers > 0,
            prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
            timeout=args.dataloader_timeout if args.workers > 0 else 0,
            drop_last=args.train_objective == "contrastive",
        )
        train_sample_count = len(train_dataset)
    if len(train_loader) == 0:
        raise RuntimeError("Training dataloader has 0 batches. Reduce --batch-size or check train data.")

    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, train_loader, args)
    scaler = None
    if args.precision == "amp" and device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    if args.resume:
        start_epoch, _, _ = load_resume(args.resume, model, optimizer, scaler)
        logger.info("Resumed checkpoint %s at epoch %d", args.resume, start_epoch)

    if args.distributed:
        dist.barrier()

    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank] if device.type == "cuda" else None)

    loss_fn = None
    if args.train_objective == "contrastive":
        loss_fn = ClipLoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
        )

    logger.info(
        "Loaded data | train_samples=%d train_classes=%d "
        "train_batches_per_rank=%d objective=%s template=%s",
        train_sample_count,
        len(train_classes),
        len(train_loader),
        args.train_objective,
        args.template,
    )

    total_start = time.time()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        if train_shared_epoch is not None:
            train_shared_epoch.set_value(epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if args.distributed:
            logger.info("Epoch %d pre-train distributed barrier started", epoch)
            dist.barrier()
            logger.info("Epoch %d pre-train distributed barrier finished", epoch)

        logger.info("Epoch %d/%d started", epoch, args.epochs)
        epoch_start = time.time()

        train_start = time.time()
        logger.info("Epoch %d train phase started", epoch)
        if args.train_objective == "text-prototype":
            train_metrics = train_one_epoch_text_prototype(
                model,
                train_loader,
                optimizer,
                scaler,
                scheduler,
                train_class_tokens,
                epoch,
                args,
                progress,
            )
        else:
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                scheduler,
                loss_fn,
                epoch,
                args,
                progress,
            )
        train_seconds = time.time() - train_start
        logger.info("Epoch %d train phase finished in %.1fs", epoch, train_seconds)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        if is_main_process(args):
            row = {"epoch": epoch, "train_loss": train_metrics["loss"],
                   "train_top1": train_metrics["top1"], "train_top5": train_metrics["top5"]}
            append_csv(output_dir / "metrics.csv", TRAIN_ONLY_METRICS_FIELDS, row)
            save_json(output_dir / "latest_metrics.json", row)
            if args.save_every_epoch_weights:
                save_epoch_weights(checkpoint_dir, model, epoch, row, args)
            # Keep optimizer/scaler state for resuming; epoch snapshots are model-only.
            temporary = checkpoint_dir / "last.tmp"
            save_checkpoint(temporary, model, optimizer, scaler, epoch, epoch,
                            0.0, args, row)
            os.replace(temporary, checkpoint_dir / "last.pt")
            logger.info("Epoch %03d finished | loss=%.6f train_seconds=%.1f",
                        epoch, train_metrics["loss"], train_seconds)

        if args.distributed:
            logger.info("Epoch %d distributed barrier started", epoch)
            dist.barrier()
            logger.info("Epoch %d distributed barrier finished", epoch)

    cleanup_distributed(args)


if __name__ == "__main__":
    main()
