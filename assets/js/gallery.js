const CASES = [
  {
    id: "trolley8",
    title: "Trolley 8",
    action:
      "The trolley is pushed by a person through the handle along the world -X axis on a road at a base transport speed of 0.6 mesh units per second, and runs for 3 seconds.",
    src: "assets/models/trolley8_push.glb",
    orientation: "0deg -90deg 0deg",
    cameraOrbit: "180deg 70deg 129%",
    fieldOfView: "35deg",
  },
  {
    id: "trolley7",
    title: "Trolley 7",
    action:
      "The trolley is pushed by a person through the handle along the world -X axis on a road at a base transport speed of 0.6 mesh units per second, and runs for 3 seconds.",
    src: "assets/models/trolley7_push.glb",
    orientation: "0deg -90deg 0deg",
    cameraOrbit: "0deg 70deg 124%",
    fieldOfView: "35deg",
  },
  {
    id: "trolley3",
    title: "Trolley 3",
    action:
      "The trolley is pushed by a person through the handle along the world -X axis on a road at a base transport speed of 0.6 mesh units per second, and runs for 3 seconds.",
    src: "assets/models/trolley3_push.glb",
    orientation: "0deg -90deg 0deg",
    cameraOrbit: "0deg 70deg 109%",
    fieldOfView: "35deg",
  },
  {
    id: "drawer8",
    title: "8 Drawer",
    action:
      "Open the masked areas(shown in translucent green area) as shown on the input mask image only for 2 seconds.",
    src: "assets/models/drawer8_agent_mask1_open.glb",
    orientation: "0deg -90deg 0deg",
    actionImage: "assets/actions/8_drawer_mask1.png",
    actionImageAlt: "8 drawer input mask",
    cameraOrbit: "270deg 70deg 95%",
    fieldOfView: "35deg",
  },
  {
    id: "kettle8",
    title: "Electric Kettle 8",
    action: "Press the button to fully open the kettle lid.",
    src: "assets/models/electric_kettle8_open_lid.glb",
    orientation: "0deg -90deg 0deg",
    cameraOrbit: "90deg 70deg 131%",
    fieldOfView: "35deg",
  },
  {
    id: "toasteroven38",
    title: "Toaster Oven 38",
    action: "A person fully opens the toaster oven door, then fully pulls out the upper tray.",
    src: "assets/models/toasteroven38_open_upper_tray.glb",
    orientation: "0deg -90deg 0deg",
    cameraOrbit: "330deg 60deg 102%",
    fieldOfView: "35deg",
  },
  {
    id: "basket1",
    title: "Basket 1",
    action:
      "A person picks up the basket handles by simultaneously rotating both handles and then lifts the basket upward for a vertical displacement of 0.3 mesh meter.",
    src: "assets/models/basket1_lift_basket.glb",
    orientation: "0deg -90deg 0deg",
    cameraOrbit: "180deg 70deg 158%",
    fieldOfView: "35deg",
  },
  {
    id: "bin1",
    title: "Trash Bin 1",
    action: "Fully open the trash bin",
    src: "assets/models/bin1_fully_open.glb",
    orientation: "0deg -90deg 0deg",
    cameraOrbit: "270deg 70deg 118%",
    fieldOfView: "35deg",
  },
  {
    id: "clock3",
    title: "Clock 3",
    action: "A clock is running 50 times faster than the reality, run for 5 seconds",
    src: "assets/models/clock3_speed_50x.glb",
    orientation: "0deg -90deg 0deg",
    cameraOrbit: "270deg 70deg 95%",
    fieldOfView: "35deg",
  },
  {
    id: "dishwasher1",
    title: "Dishwasher 1",
    action: "open the tray at the bottom of the dishwasher",
    src: "assets/models/dishwasher1_open_bottom_tray.glb",
    orientation: "0deg -90deg 0deg",
    cameraOrbit: "330deg 60deg 116%",
    fieldOfView: "35deg",
  },
  {
    id: "door4",
    title: "Door 4",
    action: "A person opens the door fully, walks through, and lets go of the handle.",
    src: "assets/models/door4_open_door.glb",
    orientation: "0deg -90deg 0deg",
    cameraOrbit: "60deg 70deg 95%",
    fieldOfView: "35deg",
  },
  {
    id: "laptop1",
    title: "Laptop Lid 1",
    action: "Take exactly 2 seconds to open the laptop lid fully.",
    src: "assets/models/laptop_lid1_open.glb",
    orientation: "0deg -90deg 0deg",
    cameraOrbit: "90deg 70deg 100%",
    fieldOfView: "35deg",
  },
  {
    id: "microwave1",
    title: "Microwave 1",
    action: "Fully Open the microwave",
    src: "assets/models/microwave1_fully_open.glb",
    orientation: "0deg -90deg 0deg",
    cameraOrbit: "0deg 70deg 120%",
    fieldOfView: "35deg",
  },
  {
    id: "mixer003",
    title: "Stand Mixer",
    action: "Set the mixing speed to level 1 and run for 3 seconds",
    src: "assets/models/mixer003_mix_speed1.glb",
    orientation: "0deg -90deg 0deg",
    cameraOrbit: "90deg 70deg 95%",
    fieldOfView: "35deg",
  },
  {
    id: "lockmixer3",
    title: "Lock Mixer 3",
    action: "Unlock and start the mixer. Set the mixing speed to level 1 and run for 3 seconds",
    src: "assets/models/lock_mixer3_unlock_mix.glb",
    orientation: "0deg -90deg 0deg",
    cameraOrbit: "90deg 70deg 95%",
    fieldOfView: "35deg",
  },
  {
    id: "ricecooker1",
    title: "Rice Cooker 1",
    action: "Fully Open the ricecooker",
    src: "assets/models/rice_cooker1_fully_open.glb",
    orientation: "0deg -90deg 0deg",
    cameraOrbit: "0deg 70deg 105%",
    fieldOfView: "35deg",
  },
  {
    id: "safe2",
    title: "Safe 2",
    action:
      "Unlock and fully open the safe. To unlock the combination lock, turn the combination dial clockwise by 90 degrees. To open the door, turn the door handle clockwise by 90 degrees first.",
    src: "assets/models/safe2_unlock_open.glb",
    orientation: "0deg -90deg 0deg",
    cameraOrbit: "270deg 70deg 96%",
    fieldOfView: "35deg",
  },
  {
    id: "scissors1",
    title: "Scissors 1",
    action: "A person opens the scissors once and then closes them.",
    src: "assets/models/scissors1_open.glb",
    orientation: "0deg -90deg 0deg",
    cameraOrbit: "90deg 70deg 95%",
    fieldOfView: "35deg",
  },
  {
    id: "table6",
    title: "Lift Table 6",
    action: "Fully raise the table.",
    src: "assets/models/table6_gt_fully_raise.glb",
    orientation: "0deg -90deg 0deg",
    cameraOrbit: "120deg 80deg 95%",
    fieldOfView: "25deg",
  },
];

const MODES = ["mesh", "action", "animation"];

function applyTheme(theme) {
  const isDark = theme === "dark";
  document.documentElement.dataset.theme = isDark ? "dark" : "light";
  document.querySelectorAll("[data-theme-toggle]").forEach((button) => {
    button.textContent = isDark ? "Day" : "Night";
    button.setAttribute("aria-label", isDark ? "Switch to day mode" : "Switch to night mode");
  });
}

const savedTheme = localStorage.getItem("artimo-theme");
const initialTheme =
  savedTheme ||
  (window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light");
applyTheme(initialTheme);

document.querySelectorAll("[data-theme-toggle]").forEach((button) => {
  button.addEventListener("click", () => {
    const nextTheme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    localStorage.setItem("artimo-theme", nextTheme);
    applyTheme(nextTheme);
  });
});

function modelViewer(item) {
  return `
    <model-viewer
      data-model-src="${item.src}"
      camera-controls
      autoplay
      touch-action="none"
      zoom-sensitivity="1.2"
      orbit-sensitivity="1"
      pan-sensitivity="1"
      animation-crossfade-duration="250"
      interaction-prompt="none"
      shadow-intensity="0.45"
      exposure="0.62"
      orientation="${item.orientation || "0deg 0deg 0deg"}"
      camera-orbit="${item.cameraOrbit}"
      field-of-view="${item.fieldOfView}"
      min-camera-orbit="auto auto 20%"
      max-camera-orbit="auto auto 800%"
      min-field-of-view="10deg"
      max-field-of-view="60deg"
    ></model-viewer>
  `;
}

function actionPanel(item) {
  const image = item.actionImage
    ? `<img class="action-image" src="${item.actionImage}" alt="${item.actionImageAlt || item.title + " action image"}" />`
    : "";

  return `
    <div class="action-panel${image ? "" : " no-image"}">
      <div class="action-copy">
        <p class="action-label">Action Prompt</p>
        <p class="action-text">${item.action}</p>
      </div>
      ${image}
    </div>
  `;
}

function modeButton(item, mode, active) {
  const label = mode.charAt(0).toUpperCase() + mode.slice(1);
  return `
    <button
      class="mode-button${active ? " is-active" : ""}"
      type="button"
      data-gallery-mode="${mode}"
      data-gallery-target="${item.id}"
      aria-pressed="${active ? "true" : "false"}"
    >
      ${label}
    </button>
  `;
}

function modelCard(item) {
  return `
    <article class="model-card" data-gallery-card="${item.id}">
      <header>
        <h3>${item.title}</h3>
        <p>${item.action}</p>
      </header>
      <div class="case-stage">
        <div class="mode-tabs" aria-label="${item.title} display mode">
          ${MODES.map((mode) => modeButton(item, mode, mode === "animation")).join("")}
        </div>
        <div class="mode-panel viewer-panel" data-gallery-panel="viewer">
          ${modelViewer(item)}
        </div>
        <div class="mode-panel" data-gallery-panel="action" hidden>
          ${actionPanel(item)}
        </div>
      </div>
    </article>
  `;
}

function setViewerMode(card, mode) {
  loadModel(card);

  const viewerPanel = card.querySelector('[data-gallery-panel="viewer"]');
  const actionPanelEl = card.querySelector('[data-gallery-panel="action"]');
  const viewer = card.querySelector("model-viewer");

  const showAction = mode === "action";
  viewerPanel.hidden = showAction;
  actionPanelEl.hidden = !showAction;

  if (!viewer) return;
  if (mode === "animation") {
    viewer.setAttribute("autoplay", "");
    viewer.play?.();
    return;
  }

  viewer.removeAttribute("autoplay");
  viewer.pause?.();
  if (mode === "mesh") {
    viewer.currentTime = 0;
  }
}

function loadModel(card) {
  const viewer = card.querySelector("model-viewer");
  if (!viewer || viewer.src || !viewer.dataset.modelSrc) return;
  viewer.src = viewer.dataset.modelSrc;
}

const gallery = document.querySelector("#asset-gallery");
if (gallery) {
  gallery.innerHTML = CASES.map(modelCard).join("");
  gallery.addEventListener("click", (event) => {
    const button = event.target.closest("[data-gallery-mode]");
    if (!button) return;

    const card = button.closest("[data-gallery-card]");
    const mode = button.dataset.galleryMode;
    if (!card || !mode) return;

    setViewerMode(card, mode);

    card.querySelectorAll("[data-gallery-mode]").forEach((tab) => {
      const isActive = tab.dataset.galleryMode === mode;
      tab.classList.toggle("is-active", isActive);
      tab.setAttribute("aria-pressed", isActive ? "true" : "false");
    });
  });

  const cards = Array.from(gallery.querySelectorAll("[data-gallery-card]"));
  cards.slice(0, 2).forEach(loadModel);

  if ("IntersectionObserver" in window) {
    const modelObserver = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            loadModel(entry.target);
            modelObserver.unobserve(entry.target);
          }
        }
      },
      {
        root: gallery,
        rootMargin: "420px",
        threshold: 0.01,
      }
    );

    cards.forEach((card) => modelObserver.observe(card));
  } else {
    cards.forEach(loadModel);
  }
}

function activateVideo(el) {
  if (!el.src && el.dataset.src) {
    el.src = el.dataset.src;
    el.load();
  }
  const playPromise = el.play();
  if (playPromise && typeof playPromise.catch === "function") {
    playPromise.catch(() => {});
  }
}

function pauseVideo(el) {
  el.pause();
}

const lazyVideos = () => Array.from(document.querySelectorAll("video.lazy-video"));

if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        const el = entry.target;
        if (entry.isIntersecting && entry.intersectionRatio > 0.35) {
          activateVideo(el);
        } else {
          pauseVideo(el);
        }
      }
    },
    {
      root: null,
      rootMargin: "160px 0px",
      threshold: [0, 0.35, 0.75],
    }
  );

  lazyVideos().forEach((el) => observer.observe(el));
} else {
  lazyVideos().slice(0, 1).forEach(activateVideo);
}
