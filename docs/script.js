const menu=document.querySelector('.menu-toggle');const nav=document.querySelector('#navigation');menu.addEventListener('click',()=>{const open=menu.getAttribute('aria-expanded')!=='true';menu.setAttribute('aria-expanded',String(open));menu.setAttribute('aria-label',open?'Close navigation':'Open navigation');nav.classList.toggle('open',open)});nav.querySelectorAll('a').forEach(a=>a.addEventListener('click',()=>{nav.classList.remove('open');menu.setAttribute('aria-expanded','false');menu.setAttribute('aria-label','Open navigation')}));const dialog=document.querySelector('#figure-dialog');const image=document.querySelector('#dialog-image');const caption=document.querySelector('#dialog-caption');function openFigure(src,text){image.src=src;image.alt=text;caption.textContent=text;dialog.showModal();document.body.classList.add('modal-open')}document.querySelectorAll('[data-image],[data-figure]').forEach(button=>button.addEventListener('click',()=>openFigure(button.dataset.image||button.dataset.figure,button.dataset.caption||'')));document.querySelector('#dialog-close').addEventListener('click',()=>dialog.close());dialog.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close()}});dialog.addEventListener('close',()=>document.body.classList.remove('modal-open'));document.querySelector('#copy-citation').addEventListener('click',async()=>{const text=document.querySelector('#citation-text').textContent;try{await navigator.clipboard.writeText(text);const toast=document.querySelector('#toast');toast.textContent='BibTeX copied';toast.classList.add('visible');setTimeout(()=>toast.classList.remove('visible'),2200)}catch{const selection=window.getSelection();const range=document.createRange();range.selectNodeContents(document.querySelector('#citation-text'));selection.removeAllRanges();selection.addRange(range);const toast=document.querySelector('#toast');toast.textContent='Citation selected — press Ctrl/Cmd+C';toast.classList.add('visible');setTimeout(()=>toast.classList.remove('visible'),3500)}});
// Switch only between bundled, recorded examples; no model requests.
document.querySelectorAll('.sample-tabs').forEach(list=>{const tabs=Array.from(list.querySelectorAll('[role="tab"]'));function activate(tab){tabs.forEach(item=>{const selected=item===tab;item.setAttribute('aria-selected',String(selected));item.tabIndex=selected?0:-1;document.getElementById(item.getAttribute('aria-controls')).hidden=!selected})}tabs.forEach((tab,index)=>{tab.addEventListener('click',()=>activate(tab));tab.addEventListener('keydown',event=>{let next=index;if(event.key==='ArrowRight')next=(index+1)%tabs.length;else if(event.key==='ArrowLeft')next=(index-1+tabs.length)%tabs.length;else if(event.key==='Home')next=0;else if(event.key==='End')next=tabs.length-1;else return;event.preventDefault();tabs[next].focus();activate(tabs[next])})});activate(tabs[0])});document.documentElement.classList.add('examples-ready');
// Keep the presentation inline; start muted only while it is in view.
const presentation = document.querySelector('.presentation-video video');
if (presentation && 'IntersectionObserver' in window) {
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let inView = false;
  presentation.muted = true;
  function updatePlayback() {
    if (!inView || document.hidden) {
      presentation.pause();
    } else if (!reducedMotion.matches) {
      // Autoplay restrictions may still apply; native controls remain usable.
      presentation.play().catch(() => {});
    }
  }
  const observer = new IntersectionObserver(([entry]) => {
    inView = entry.isIntersecting && entry.intersectionRatio >= 0.5;
    updatePlayback();
  }, {threshold: [0, 0.5]});
  observer.observe(presentation);
  document.addEventListener('visibilitychange', updatePlayback);
  reducedMotion.addEventListener('change', () => {
    if (reducedMotion.matches) presentation.pause();
    else updatePlayback();
  });
}
