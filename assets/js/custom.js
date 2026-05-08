document.addEventListener("DOMContentLoaded", () => {
    // ==========================================
    // 1. Article Headings Collapse (H1 - H6)
    // ==========================================
    const postContent = document.querySelector('.post-content');
    if (postContent) {
        const headers = postContent.querySelectorAll('h1, h2, h3, h4, h5, h6');
        
        headers.forEach(header => {
            header.style.cursor = 'pointer';
            header.title = 'Click to toggle';
            
            const indicator = document.createElement('span');
            indicator.innerHTML = ' &#9660;'; // Downward triangle
            indicator.style.fontSize = '0.6em';
            indicator.style.marginLeft = '8px';
            indicator.style.opacity = '0.5';
            indicator.style.display = 'inline-block';
            indicator.style.transition = 'transform 0.2s';
            header.appendChild(indicator);

            header.addEventListener('click', (e) => {
                if (e.target.tagName.toLowerCase() === 'a') return;
                
                const level = parseInt(header.tagName.substring(1));
                const isCollapsed = header.classList.toggle('is-collapsed');
                
                if (isCollapsed) {
                    indicator.style.transform = 'rotate(-90deg)';
                } else {
                    indicator.style.transform = 'rotate(0deg)';
                }

                let current = header.nextElementSibling;
                while (current) {
                    if (current.tagName.match(/^H[1-6]$/)) {
                        const currentLevel = parseInt(current.tagName.substring(1));
                        if (currentLevel <= level) {
                            break;
                        }
                    }
                    
                    if (isCollapsed) {
                        if (!current.hasAttribute('data-original-display')) {
                            current.setAttribute('data-original-display', current.style.display || '');
                        }
                        current.style.display = 'none';
                    } else {
                        current.style.display = current.getAttribute('data-original-display') || '';
                    }
                    
                    current = current.nextElementSibling;
                }
            });
        });
    }

    // ==========================================
    // 2. TOC (Table of Contents) Collapse
    // ==========================================
    const toc = document.querySelector('#TableOfContents');
    if (toc) {
        // Find all list items that contain a nested list
        const tocItemsWithChildren = toc.querySelectorAll('li');
        
        tocItemsWithChildren.forEach(li => {
            const childUl = li.querySelector('ul');
            if (childUl) {
                // This step has children, add a toggle indicator
                const wrapper = document.createElement('div');
                wrapper.style.position = 'relative';
                
                const indicator = document.createElement('span');
                indicator.innerHTML = '&#9660;';
                indicator.style.cursor = 'pointer';
                indicator.style.fontSize = '0.8em';
                indicator.style.position = 'absolute';
                indicator.style.left = '-18px';
                indicator.style.top = '7px';
                indicator.style.opacity = '0.6';
                indicator.style.transition = 'transform 0.2s';
                
                // Wrap the contents of LI up to the UL? 
                // Wait, it's easier to insert it into the LI
                li.style.position = 'relative';
                li.insertBefore(indicator, li.firstChild);
                
                indicator.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const isCollapsed = li.classList.toggle('toc-collapsed');
                    if (isCollapsed) {
                        indicator.style.transform = 'rotate(-90deg)';
                        childUl.style.display = 'none';
                    } else {
                        indicator.style.transform = 'rotate(0deg)';
                        childUl.style.display = 'block';
                    }
                });
            }
        });
    }
});
