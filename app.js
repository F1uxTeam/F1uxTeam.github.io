const content = window.F1UX_CONTENT || {};
const posts = (content.posts || []).map((post, index) => ({ ...post, index }));
const members = (content.members || []).map((member, index) => ({ ...member, index }));
const memberOrder = shuffledIndexes(members.length);
const memberPosition = new Map(memberOrder.map((memberIndex, position) => [memberIndex, position]));
const page = document.body.dataset.page || "home";

const state = {
  category: "All",
  query: "",
  sort: "new",
};

const memberState = {
  role: "All",
  query: "",
  spotlight: memberOrder[0] ?? 0,
};

function $(selector) {
  return document.querySelector(selector);
}

function $all(selector) {
  return [...document.querySelectorAll(selector)];
}

function shuffledIndexes(length) {
  const indexes = Array.from({ length }, (_, index) => index);
  for (let i = indexes.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [indexes[i], indexes[j]] = [indexes[j], indexes[i]];
  }
  return indexes;
}

function escapeHTML(value = "") {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatDate(value) {
  return new Intl.DateTimeFormat("en", {
    month: "short",
    day: "2-digit",
    year: "numeric",
  }).format(new Date(`${value}T00:00:00`));
}

function categoryClass(category) {
  return String(category || "writeup")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-");
}

function trackSummary(post) {
  return (post.tracks?.length ? post.tracks : ["Writeup"]).slice(0, 4).join(" · ");
}

function refreshIcons() {
  if (window.lucide) {
    window.lucide.createIcons();
  }
}

function setText(selector, value) {
  const element = $(selector);
  if (element) element.textContent = value;
}

function setActiveNav() {
  $all("[data-nav]").forEach((link) => {
    link.classList.toggle("is-active", link.dataset.nav === page);
  });
}

function setupTheme() {
  const themeToggle = $("#themeToggle");
  if (!themeToggle) return;

  if (localStorage.getItem("f1ux-theme") === "dim") {
    document.body.classList.add("is-dim");
    themeToggle.innerHTML = '<i data-lucide="moon-star"></i>';
  }

  themeToggle.addEventListener("click", () => {
    document.body.classList.toggle("is-dim");
    const isDim = document.body.classList.contains("is-dim");
    localStorage.setItem("f1ux-theme", isDim ? "dim" : "light");
    themeToggle.innerHTML = isDim ? '<i data-lucide="moon-star"></i>' : '<i data-lucide="sun-medium"></i>';
    refreshIcons();
  });
}

function renderPostCard(post, variant = "") {
  return `
    <article class="post-card glass-surface motion-card ${variant}" data-post-link="${post.index}" style="--card-accent: ${escapeHTML(post.accent)}">
      <div class="post-card-header">
        <span class="category-pill">
          <span class="category-dot ${categoryClass(post.category)}"></span>
          ${escapeHTML(post.category)}
        </span>
        <span class="difficulty">${escapeHTML(post.difficulty)}</span>
      </div>
      <h3>${escapeHTML(post.title)}</h3>
      <p>${escapeHTML(post.excerpt)}</p>
      <div class="article-meta">
        <span><i data-lucide="calendar-days"></i>${formatDate(post.date)}</span>
        <span><i data-lucide="clock-3"></i>${escapeHTML(post.readTime)}</span>
      </div>
      <div class="tag-row">
        <span>${escapeHTML(post.event || post.category || "CTF")}</span>
      </div>
    </article>
  `;
}

function renderHome() {
  const latest = posts[0];
  const tracks = [...new Set(posts.flatMap((post) => post.tracks || []))];

  setText("#postMetric", posts.length);
  setText("#trackMetric", tracks.length);
  setText("#memberMetric", members.length);

  const spotlight = $("#homeSpotlight");
  if (latest && spotlight) {
    spotlight.dataset.postLink = latest.index;
    spotlight.innerHTML = `
      <span class="category-pill"><span class="category-dot ${categoryClass(latest.category)}"></span>${escapeHTML(latest.event)}</span>
      <h2>${escapeHTML(latest.title)}</h2>
      <p>${escapeHTML(latest.excerpt)}</p>
      <div class="article-meta">
        <span><i data-lucide="calendar-days"></i>${formatDate(latest.date)}</span>
        <span><i data-lucide="clock-3"></i>${escapeHTML(latest.readTime)}</span>
      </div>
    `;
  }

  const trackRiver = $("#trackRiver");
  if (trackRiver) {
    trackRiver.innerHTML = tracks
      .slice(0, 9)
      .map((track, index) => `<span style="--delay: ${index * 90}ms">${escapeHTML(track)}</span>`)
      .join("");
  }

  const homePostGrid = $("#homePostGrid");
  if (homePostGrid) {
    homePostGrid.innerHTML = posts.slice(0, 3).map((post) => renderPostCard(post, "home-post")).join("");
  }

  const reel = $("#homeMemberReel");
  if (reel) {
    reel.innerHTML = members
      .slice(0, 12)
      .map(
        (member, index) => `
          <a class="reel-member" href="/members/" style="--offset: ${index % 4}">
            <img src="${escapeHTML(member.avatar)}" alt="${escapeHTML(member.name)}" loading="lazy" onerror="this.onerror=null;this.src='/assets/f1ux-logo.png';" />
            <span>${escapeHTML(member.name)}</span>
          </a>
        `,
      )
      .join("");
  }
}

function postMatches(post) {
  const haystack = [
    post.title,
    post.category,
    post.author,
    post.event,
    post.excerpt,
    post.sourceFile,
    ...(post.tags || []),
    ...(post.tracks || []),
    post.markdown || "",
  ]
    .join(" ")
    .toLowerCase();

  const categoryMatch = state.category === "All" || post.category === state.category;
  const queryMatch = haystack.includes(state.query.trim().toLowerCase());
  return categoryMatch && queryMatch;
}

function getVisiblePosts() {
  const visible = posts.filter(postMatches);
  if (state.sort === "hot") {
    return visible.sort((a, b) => b.hot - a.hot);
  }
  return visible.sort((a, b) => new Date(b.date) - new Date(a.date));
}

function renderCategories() {
  const categoryTabs = $("#categoryTabs");
  if (!categoryTabs) return;

  const categories = ["All", ...new Set(posts.map((post) => post.category))];
  categoryTabs.innerHTML = categories
    .map(
      (category) => `
        <button class="category-tab ${state.category === category ? "is-active" : ""}" type="button" data-category="${escapeHTML(category)}">
          ${escapeHTML(category)}
        </button>
      `,
    )
    .join("");
}

function renderPosts() {
  const postGrid = $("#postGrid");
  if (!postGrid) return;

  const visible = getVisiblePosts();
  setText("#resultCount", `${visible.length} posts`);
  postGrid.innerHTML = visible.map((post) => renderPostCard(post)).join("");

  if (!visible.length) {
    postGrid.innerHTML = `
      <article class="post-card glass-surface">
        <div class="post-card-header">
          <span class="category-pill"><span class="category-dot"></span>No result</span>
        </div>
        <h3>没有匹配的文章</h3>
        <p>换一个关键词或分类即可刷新列表。</p>
      </article>
    `;
  }

  refreshIcons();
}

function renderArchiveDeck() {
  const archiveDeck = $("#archiveDeck");
  if (!archiveDeck) return;

  const grouped = posts.reduce((acc, post) => {
    const year = String(new Date(`${post.date}T00:00:00`).getFullYear());
    acc[year] ||= [];
    acc[year].push(post);
    return acc;
  }, {});

  archiveDeck.innerHTML = Object.entries(grouped)
    .sort(([a], [b]) => Number(b) - Number(a))
    .map(
      ([year, yearPosts]) => `
        <section class="archive-year glass-surface">
          <div class="archive-year-head">
            <span>${escapeHTML(year)}</span>
            <strong>${yearPosts.length} posts</strong>
          </div>
          <div class="archive-list">
            ${yearPosts
              .sort((a, b) => new Date(b.date) - new Date(a.date))
              .map(
                (post) => `
                  <button class="archive-row archive-row-rich" type="button" data-post-link="${post.index}">
                    <span class="archive-date">${formatDate(post.date)}</span>
                    <strong>${escapeHTML(post.title)}</strong>
                    <span>${escapeHTML(trackSummary(post))}</span>
                  </button>
                `,
              )
              .join("")}
          </div>
        </section>
      `,
    )
    .join("");
}

function memberMatches(member) {
  const roleMatch = memberState.role === "All" || member.role === memberState.role;
  const haystack = [member.name, member.role, memberRoleLabel(member.role), member.bio, member.url].join(" ").toLowerCase();
  const queryMatch = haystack.includes(memberState.query.trim().toLowerCase());
  return roleMatch && queryMatch;
}

function getVisibleMembers() {
  return members
    .filter(memberMatches)
    .sort((a, b) => (memberPosition.get(a.index) ?? a.index) - (memberPosition.get(b.index) ?? b.index));
}

function memberCoordinates(index, total) {
  if (total <= 1) return { x: 50, y: 48 };

  const angle = (Math.PI * 2 * index) / total - Math.PI / 2;
  const ring = [0.98, 0.68, 0.44][index % 3];
  const x = 50 + Math.cos(angle) * 38 * ring;
  const y = 50 + Math.sin(angle) * 33 * ring;

  return {
    x: Math.min(88, Math.max(12, x)),
    y: Math.min(86, Math.max(14, y)),
  };
}

function memberAccent(member, index) {
  if (member.role === "Leader") return "var(--amber)";
  return ["var(--cyan)", "var(--mint)", "var(--blue)", "var(--coral)"][index % 4];
}

function memberRoleLabel(role) {
  return role === "Leader" ? "Captain" : role;
}

function memberInitial(name = "F") {
  return [...String(name).trim()][0]?.toUpperCase() || "F";
}

function renderMemberLens(index = memberState.spotlight) {
  const lens = $("#memberLens");
  if (!lens || !members.length) return;

  const member = members.find((item) => item.index === Number(index)) || getVisibleMembers()[0] || members[0];
  memberState.spotlight = member.index;

  lens.innerHTML = `
    <div class="lens-topline">
      <span>F1UX SIGNAL</span>
      <span>${escapeHTML(memberRoleLabel(member.role))}</span>
    </div>
    <div class="lens-avatar-wrap">
      <img src="${escapeHTML(member.avatar)}" alt="${escapeHTML(member.name)}" loading="lazy" onerror="this.onerror=null;this.src='/assets/f1ux-logo.png';" />
    </div>
    <div class="lens-copy">
      <span class="lens-initial">${escapeHTML(memberInitial(member.name))}</span>
      <h2>${escapeHTML(member.name)}</h2>
      <p>${escapeHTML(member.bio || "F1ux member")}</p>
    </div>
    <div class="lens-footer">
      <a class="primary-action profile-link" href="${escapeHTML(member.url || "#")}" target="_blank" rel="noopener noreferrer">
        <span>Profile</span>
        <i data-lucide="arrow-up-right"></i>
      </a>
    </div>
  `;

  $all("[data-member-link]").forEach((node) => {
    node.classList.toggle("is-active", Number(node.dataset.memberLink) === member.index);
  });

  refreshIcons();
}

function renderMembers() {
  const memberDirectory = $("#memberDirectory");
  if (!memberDirectory) return;

  const visible = getVisibleMembers();
  setText("#memberCount", `${visible.length} members`);

  if (!visible.length) {
    memberDirectory.innerHTML = `
      <div class="empty-directory">
        <strong>没有匹配成员</strong>
        <p>换一个关键词或角色筛选。</p>
      </div>
    `;
    const lens = $("#memberLens");
    if (lens) lens.innerHTML = "";
    return;
  }

  if (!visible.some((member) => member.index === memberState.spotlight)) {
    memberState.spotlight = visible[0].index;
  }

  memberDirectory.innerHTML = `
    <div class="signal-deck">
      ${visible
        .map((member, index) => {
          const accent = memberAccent(member, index);
          const lift = [0, 7, -3, 4, -6, 2][index % 6];
          return `
            <button class="signal-card motion-card ${member.index === memberState.spotlight ? "is-active" : ""}" type="button" data-member-link="${member.index}" style="--accent: ${accent}; --tilt: ${((index % 5) - 2) * 0.7}deg; --lift: ${lift}px;">
              <span class="signal-card-glow" aria-hidden="true"></span>
              <span class="signal-avatar">
                <img src="${escapeHTML(member.avatar)}" alt="${escapeHTML(member.name)}" loading="lazy" onerror="this.onerror=null;this.src='/assets/f1ux-logo.png';" />
              </span>
              <span class="signal-copy">
                <strong>${escapeHTML(member.name)}</strong>
                <span>${escapeHTML(member.bio || "F1ux member")}</span>
              </span>
              <span class="signal-role">${escapeHTML(memberRoleLabel(member.role))}</span>
              <span class="signal-bars" aria-hidden="true"><i></i><i></i><i></i></span>
              <span class="signal-initial" aria-hidden="true">${escapeHTML(memberInitial(member.name))}</span>
            </button>
          `;
        })
        .join("")}
    </div>
  `;

  renderMemberLens(memberState.spotlight);
  refreshIcons();
}

function renderMarkdown(markdown) {
  const source = markdown || "";
  if (window.marked?.parse) {
    return window.marked.parse(source, { gfm: true, breaks: false });
  }
  if (typeof window.marked === "function") {
    return window.marked(source);
  }
  return `<pre><code>${escapeHTML(source)}</code></pre>`;
}

function headingSlug(text, index) {
  const slug = String(text || "")
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fa5]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
  return `section-${index}-${slug || "heading"}`;
}

function buildReaderToc(readerContent) {
  const toc = $("#readerToc");
  if (!toc || !readerContent) return;

  const panel = readerContent.closest(".reader-panel");
  const headings = [...readerContent.querySelectorAll(".reader-body h1, .reader-body h2, .reader-body h3, .reader-body h4")];
  if (!headings.length) {
    panel?.classList.remove("has-toc");
    toc.hidden = true;
    toc.innerHTML = "";
    return;
  }

  panel?.classList.add("has-toc");
  toc.hidden = false;
  const used = new Set();
  toc.innerHTML = headings
    .map((heading, index) => {
      let id = heading.id || headingSlug(heading.textContent, index);
      while (used.has(id)) id = `${id}-${used.size}`;
      used.add(id);
      heading.id = id;
      const level = Number(heading.tagName.replace("H", ""));
      return `
        <button class="toc-link level-${level}" type="button" data-toc-target="${escapeHTML(id)}">
          ${escapeHTML(heading.textContent.trim())}
        </button>
      `;
    })
    .join("");
}

function enhanceMarkdownContainer(container) {
  if (!container) return;
  container.querySelectorAll("a[href]").forEach((anchor) => {
    const href = anchor.getAttribute("href") || "";
    if (/^https?:\/\//.test(href)) {
      anchor.target = "_blank";
      anchor.rel = "noopener noreferrer";
    }
  });
  container.querySelectorAll("img").forEach((image) => {
    image.loading = "lazy";
    image.onerror = () => {
      image.onerror = null;
      image.src = "/assets/f1ux-logo.png";
    };
  });
}

function enhanceReaderBody() {
  const readerContent = $("#readerContent");
  enhanceMarkdownContainer(readerContent);
}

function openReader(index) {
  const reader = $("#reader");
  const readerContent = $("#readerContent");
  const post = posts.find((item) => item.index === Number(index));
  if (!reader || !readerContent || !post) return;

  readerContent.innerHTML = `
    <span class="reader-category">
      <span class="category-dot ${categoryClass(post.category)}"></span>
      ${escapeHTML(post.category)}
    </span>
    <h2 id="readerTitle">${escapeHTML(post.title)}</h2>
    <div class="article-meta">
      <span><i data-lucide="user-round"></i>${escapeHTML(post.author)}</span>
      <span><i data-lucide="calendar-days"></i>${formatDate(post.date)}</span>
      <span><i data-lucide="clock-3"></i>${escapeHTML(post.readTime)}</span>
      <span><i data-lucide="layers-3"></i>${escapeHTML(trackSummary(post))}</span>
    </div>
    <div class="reader-tags">
      ${(post.tags || []).map((tag) => `<span>#${escapeHTML(tag)}</span>`).join("")}
    </div>
    <div class="reader-body">${renderMarkdown(post.markdown)}</div>
  `;

  enhanceReaderBody();
  buildReaderToc(readerContent);
  reader.classList.add("is-open");
  reader.setAttribute("aria-hidden", "false");
  document.body.style.overflow = "hidden";
  $(".reader-panel").scrollTop = 0;
  refreshIcons();
}

function closeReader() {
  const reader = $("#reader");
  if (!reader) return;
  reader.classList.remove("is-open");
  reader.setAttribute("aria-hidden", "true");
  document.body.style.overflow = "";
}

function setupInteractions() {
  $("#categoryTabs")?.addEventListener("click", (event) => {
    const button = event.target.closest("[data-category]");
    if (!button) return;
    state.category = button.dataset.category;
    renderCategories();
    renderPosts();
  });

  $("#searchInput")?.addEventListener("input", (event) => {
    state.query = event.target.value;
    renderPosts();
  });

  $all(".sort-option").forEach((button) => {
    button.addEventListener("click", () => {
      state.sort = button.dataset.sort;
      $all(".sort-option").forEach((option) => option.classList.remove("is-active"));
      button.classList.add("is-active");
      renderPosts();
    });
  });

  $("#memberSearch")?.addEventListener("input", (event) => {
    memberState.query = event.target.value;
    renderMembers();
  });

  $("#memberRoleTabs")?.addEventListener("click", (event) => {
    const button = event.target.closest("[data-role]");
    if (!button) return;
    memberState.role = button.dataset.role;
    $all("[data-role]").forEach((item) => item.classList.toggle("is-active", item === button));
    renderMembers();
  });

  document.addEventListener("click", (event) => {
    const shuffle = event.target.closest("#shuffleMember");
    if (shuffle) {
      event.stopPropagation();
      const visible = getVisibleMembers();
      if (visible.length) {
        const currentPosition = visible.findIndex((member) => member.index === memberState.spotlight);
        const next = visible[(currentPosition + 1 + Math.floor(Math.random() * Math.max(1, visible.length - 1))) % visible.length];
        renderMemberLens(next.index);
      }
      return;
    }

    const tocLink = event.target.closest("[data-toc-target]");
    if (tocLink) {
      const target = document.getElementById(tocLink.dataset.tocTarget);
      if (target) target.scrollIntoView({ block: "start", behavior: "smooth" });
      $all(".toc-link").forEach((link) => link.classList.toggle("is-active", link === tocLink));
      return;
    }

    const postLink = event.target.closest("[data-post-link]");
    if (postLink) {
      openReader(postLink.dataset.postLink);
      return;
    }

    const memberLink = event.target.closest("[data-member-link]");
    if (memberLink) {
      renderMemberLens(memberLink.dataset.memberLink);
      return;
    }

    if (event.target.closest("[data-reader-close]")) {
      closeReader();
      return;
    }

  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeReader();
    }
  });

  document.addEventListener("pointermove", (event) => {
    const surface = event.target.closest(".motion-card, .about-hero, .honor-board, .honor-board-aside, .honor-record, .member-lens, .reader-panel");
    if (!surface) return;
    const rect = surface.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width - 0.5;
    const y = (event.clientY - rect.top) / rect.height - 0.5;
    surface.style.setProperty("--mx", `${((x + 0.5) * 100).toFixed(1)}%`);
    surface.style.setProperty("--my", `${((y + 0.5) * 100).toFixed(1)}%`);
    if (surface.classList.contains("motion-card")) {
      surface.style.setProperty("--rx", `${(-y * 4).toFixed(2)}deg`);
      surface.style.setProperty("--ry", `${(x * 5).toFixed(2)}deg`);
    }
  });

  document.addEventListener("pointerleave", (event) => {
    const surface = event.target.closest?.(".motion-card, .about-hero, .honor-board, .honor-board-aside, .honor-record, .member-lens, .reader-panel");
    if (!surface) return;
    surface.style.removeProperty("--mx");
    surface.style.removeProperty("--my");
    surface.style.setProperty("--rx", "0deg");
    surface.style.setProperty("--ry", "0deg");
  }, true);
}

function init() {
  setActiveNav();
  setupTheme();
  setupInteractions();

  if (page === "home") {
    renderHome();
  }

  if (page === "writeups") {
    renderCategories();
    renderPosts();
  }

  if (page === "archive") {
    renderArchiveDeck();
  }

  if (page === "members") {
    renderMembers();
  }

  refreshIcons();
}

init();
