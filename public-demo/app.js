/* Route Lab UI: every path and metric comes from the public synthetic engine. */
(function (root) {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const engine = root.NFLTrajectoryEngine;
  const data = root.NFL_TRAJECTORY_DATA;
  const NS = "http://www.w3.org/2000/svg";
  const METHODS = [
    "constant_velocity",
    "hold_last_position",
    "damped_velocity",
  ];
  const DESCRIPTIONS = {
    constant_velocity:
      "Extend the last observed displacement at the same speed and direction.",
    hold_last_position:
      "Keep each track at its final observed coordinates. No future movement is assumed.",
    damped_velocity:
      "Reduce the last observed displacement at every future step. No parameter is fitted.",
  };
  const COLORS = {
    selected: "#a76019",
    constant_velocity: "#3767a3",
    hold_last_position: "#65715e",
  };
  const state = {
    game: 105,
    play: 1,
    player: "all",
    method: "damped_velocity",
    horizon: 12,
    damping: 0.9,
    frame: 12,
    truth: false,
    view: "routes",
  };
  const reducedMotion = Boolean(
    root.matchMedia &&
      root.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  let runs = null;
  let fullVelocity = null;
  let selectedMetrics = null;
  let perTrack = [];
  let timeline = [];
  let playing = false;
  let animation = null;
  let startTime = null;
  let startFrame = 0;

  function node(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = String(text);
    return element;
  }
  function svg(tag, attributes, text) {
    const element = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attributes || {}))
      element.setAttribute(key, String(value));
    if (text !== undefined) element.textContent = String(text);
    return element;
  }
  function fontSize(element, width, rendered = 12) {
    const actual = element.getBoundingClientRect?.().width;
    return actual > 0
      ? Math.max(rendered, (rendered * width) / actual)
      : rendered;
  }
  function trackName(id) {
    return `Track ${String.fromCharCode(64 + Number(id))}`;
  }
  function samePlay(row) {
    return row.game_id === state.game && row.play_id === state.play;
  }
  function chosen(row, player = state.player) {
    return samePlay(row) && (player === "all" || row.nfl_id === Number(player));
  }
  function rowsFor(rows, player = state.player) {
    return rows.filter(
      (row) => chosen(row, player) && row.frame_id <= state.horizon,
    );
  }
  function yards(value) {
    return `${value.toFixed(4)} yd`;
  }
  function playTitle() {
    return `Synthetic game ${state.game} · Play ${String(state.play).padStart(2, "0")}`;
  }
  function evaluateRows(predictions, player = state.player) {
    return engine.evaluate(
      rowsFor(predictions, player),
      rowsFor(data.labels, player),
    );
  }
  function updateControls() {
    $("horizon-value").textContent =
      `${(state.horizon / 10).toFixed(1)} s / ${state.horizon} frame${state.horizon === 1 ? "" : "s"}`;
    $("damping-value").textContent = state.damping.toFixed(2);
    $("damping").disabled = state.method !== "damped_velocity";
    $("damping-help").textContent =
      state.method === "damped_velocity"
        ? "1.00 matches constant velocity. 0.00 holds the last position."
        : "This control applies only to the damped-velocity rule.";
    $("method-description").textContent =
      DESCRIPTIONS[state.method] || "Choose a supported forecast rule.";
    $("frame").max = String(state.horizon);
    $("frame").value = String(state.frame);
    $("frame-value").textContent = `${state.frame} / ${state.horizon}`;
    $("field-clock").textContent = `t + ${(state.frame / 10).toFixed(1)} s`;
    $("field-title").textContent = playTitle();
    $("truth-legend").hidden = !state.truth;
    $("play").textContent = playing ? "Pause Ⅱ" : "Play ▶";
    $("play").setAttribute(
      "aria-label",
      playing ? "Pause forecast animation" : "Play forecast animation",
    );
    $("play").disabled = reducedMotion || !runs;
    $("previous").disabled = state.frame === 0 || !runs;
    $("next").disabled = state.frame === state.horizon || !runs;
    $("motion-note").textContent = reducedMotion
      ? "Reduced-motion preference detected. Use the frame slider or step buttons to inspect the paths."
      : "Scrub to inspect any frame. Playback advances at the data’s 10 Hz rate.";
    $("field-summary").textContent =
      `${state.player === "all" ? "3 anonymous tracks" : trackName(state.player) + " highlighted"} · coordinates in yards`;
  }
  function fieldBounds() {
    if (state.view === "field") return { x0: 0, x1: 120, y0: 0, y1: 53.333 };
    // Crop derives only from observations and a full CV forecast, never hidden truth.
    const rows = [
      ...data.observations.filter(samePlay),
      ...fullVelocity.predictions.filter(samePlay),
    ];
    const xs = rows.map((row) => row.x),
      ys = rows.map((row) => row.y);
    const centerX = (Math.min(...xs) + Math.max(...xs)) / 2;
    const centerY = (Math.min(...ys) + Math.max(...ys)) / 2;
    const spanY = Math.max(
      15,
      Math.max(...ys) - Math.min(...ys) + 8,
      ((Math.max(...xs) - Math.min(...xs) + 8) * 402) / 700,
    );
    const spanX = (spanY * 700) / 402;
    const x0 = Math.max(0, Math.min(120 - spanX, centerX - spanX / 2));
    const y0 = Math.max(0, Math.min(53.333 - spanY, centerY - spanY / 2));
    return { x0, x1: x0 + spanX, y0, y1: y0 + spanY };
  }
  function renderField() {
    if (!runs) return;
    const field = $("field"),
      bounds = fieldBounds(),
      type = fontSize(field, 760);
    const fieldHeight = state.view === "field" ? (700 * 53.333) / 120 : 402;
    const box = {
      left: 40,
      top: 18 + (402 - fieldHeight) / 2,
      width: 700,
      height: fieldHeight,
    };
    const x = (value) =>
      box.left + ((value - bounds.x0) / (bounds.x1 - bounds.x0)) * box.width;
    const y = (value) =>
      box.top + ((value - bounds.y0) / (bounds.y1 - bounds.y0)) * box.height;
    const points = (rows) =>
      rows
        .map((row) => `${x(row.x).toFixed(3)},${y(row.y).toFixed(3)}`)
        .join(" ");
    field.replaceChildren();
    const defs = svg("defs"),
      clip = svg("clipPath", { id: "route-field-clip" });
    clip.append(
      svg("rect", {
        x: box.left,
        y: box.top,
        width: box.width,
        height: box.height,
        rx: 3,
      }),
    );
    defs.append(clip);
    field.append(defs);
    const content = svg("g", { "clip-path": "url(#route-field-clip)" });
    content.append(
      svg("rect", {
        x: box.left,
        y: box.top,
        width: box.width,
        height: box.height,
        fill: "#173d2f",
      }),
    );
    for (let yard = 0; yard < 120; yard += 10) {
      content.append(
        svg("rect", {
          x: x(yard),
          y: y(0),
          width: x(yard + 10) - x(yard),
          height: y(53.333) - y(0),
          fill: yard % 20 === 0 ? "#1a4333" : "#173d2f",
        }),
      );
    }
    for (let yard = 0; yard <= 120; yard += 5) {
      content.append(
        svg("line", {
          x1: x(yard),
          y1: y(0),
          x2: x(yard),
          y2: y(53.333),
          stroke: "#9cb096",
          "stroke-width": yard % 10 === 0 ? 1.2 : 0.7,
          opacity: yard % 10 === 0 ? 0.6 : 0.28,
        }),
      );
    }
    for (let yard = 10; yard <= 110; yard++) {
      for (const lateral of [17.78, 35.55])
        content.append(
          svg("line", {
            x1: x(yard),
            y1: y(lateral - 0.3),
            x2: x(yard),
            y2: y(lateral + 0.3),
            stroke: "#a7b89d",
            "stroke-width": 1,
            opacity: 0.55,
          }),
        );
    }
    content.append(
      svg("rect", {
        x: x(0),
        y: y(0),
        width: x(120) - x(0),
        height: y(53.333) - y(0),
        fill: "none",
        stroke: "#d6dec9",
        "stroke-width": 2,
      }),
    );
    if (state.view === "field") {
      for (let yard = 20; yard <= 100; yard += 10) {
        const number = 50 - Math.abs(yard - 60);
        for (const lateral of [7, 47])
          content.append(
            svg(
              "text",
              {
                x: x(yard),
                y: y(lateral),
                fill: "#c7d4bd",
                opacity: 0.8,
                "font-size": type * 1.25,
                "text-anchor": "middle",
                "font-family": "Georgia,serif",
              },
              number,
            ),
          );
      }
    }
    for (const id of [1, 2, 3]) {
      const history = data.observations
        .filter((row) => samePlay(row) && row.nfl_id === id)
        .sort((a, b) => a.frame_id - b.frame_id);
      const last = history.at(-1);
      const forecast = runs[state.method].predictions
        .filter(
          (row) =>
            samePlay(row) && row.nfl_id === id && row.frame_id <= state.frame,
        )
        .sort((a, b) => a.frame_id - b.frame_id);
      const opacity =
        state.player === "all" || Number(state.player) === id ? 1 : 0.3;
      const group = svg("g", { opacity });
      group.append(
        svg("polyline", {
          points: points(history),
          fill: "none",
          stroke: "#e3edd4",
          "stroke-width": 3,
          "stroke-linejoin": "round",
          "stroke-linecap": "round",
        }),
      );
      for (const row of history)
        group.append(
          svg("circle", {
            cx: x(row.x),
            cy: y(row.y),
            r: 2.5,
            fill: "#e3edd4",
          }),
        );
      if (forecast.length) {
        group.append(
          svg("polyline", {
            points: points([last, ...forecast]),
            fill: "none",
            stroke: "#efbe70",
            "stroke-width": 3,
            "stroke-dasharray": "7 5",
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
          }),
        );
        for (const row of forecast)
          group.append(
            svg("circle", {
              cx: x(row.x),
              cy: y(row.y),
              r: 2.1,
              fill: "#efbe70",
            }),
          );
      }
      if (state.truth) {
        const future = data.labels
          .filter(
            (row) =>
              samePlay(row) && row.nfl_id === id && row.frame_id <= state.frame,
          )
          .sort((a, b) => a.frame_id - b.frame_id);
        if (future.length) {
          group.append(
            svg("polyline", {
              points: points([last, ...future]),
              fill: "none",
              stroke: "#b7d5ff",
              "stroke-width": 2.5,
              "stroke-dasharray": "2 5",
              "stroke-linecap": "round",
            }),
          );
          const end = future.at(-1),
            cx = x(end.x),
            cy = y(end.y);
          group.append(
            svg("path", {
              d: `M${cx},${cy - 5} L${cx + 5},${cy} L${cx},${cy + 5} L${cx - 5},${cy} Z`,
              fill: "#b7d5ff",
            }),
          );
        }
      }
      const endpoint = forecast.at(-1) || last;
      const marker = svg("circle", {
        cx: x(endpoint.x),
        cy: y(endpoint.y),
        r: 6,
        fill: "#efbe70",
        stroke: "#173d2f",
        "stroke-width": 2,
      });
      marker.append(
        svg(
          "title",
          {},
          `${trackName(id)}: x ${endpoint.x.toFixed(3)}, y ${endpoint.y.toFixed(3)} yards at future frame ${state.frame}`,
        ),
      );
      group.append(marker);
      group.append(
        svg(
          "text",
          {
            x: x(endpoint.x) + 10,
            y: y(endpoint.y) - 10,
            fill: "#fff5dd",
            "font-size": type * 1.1,
            "font-family": "ui-monospace,monospace",
            "font-weight": 650,
          },
          String.fromCharCode(64 + id),
        ),
      );
      content.append(group);
    }
    field.append(content);
    const increment = state.view === "field" ? 20 : 5;
    for (
      let yard = Math.ceil(bounds.x0 / increment) * increment;
      yard <= bounds.x1;
      yard += increment
    ) {
      if (x(yard) < box.left + type || x(yard) > box.left + box.width - type)
        continue;
      field.append(
        svg(
          "text",
          {
            x: x(yard),
            y: box.top + box.height + type * 1.8,
            fill: "#d1dec7",
            "font-size": type,
            "text-anchor": "middle",
            "font-family": "ui-monospace,monospace",
          },
          yard,
        ),
      );
    }
    for (
      let yard = Math.ceil(bounds.y0 / 10) * 10;
      yard <= bounds.y1;
      yard += 10
    ) {
      if (y(yard) < box.top + type || y(yard) > box.top + box.height - type)
        continue;
      field.append(
        svg(
          "text",
          {
            x: box.left - 9,
            y: y(yard) + type * 0.3,
            fill: "#d1dec7",
            "font-size": type,
            "text-anchor": "end",
            "font-family": "ui-monospace,monospace",
          },
          yard,
        ),
      );
    }
    field.setAttribute(
      "aria-label",
      `${playTitle()}, ${state.player === "all" ? "all three tracks" : trackName(state.player)}, ${state.method.replaceAll("_", " ")}, future frame ${state.frame} of ${state.horizon}. Solid observed paths, dashed forecast. Future truth ${state.truth ? "shown as dotted blue paths" : "hidden"}. x and y coordinates in yards.`,
    );
  }
  function calculateMetrics() {
    selectedMetrics = Object.fromEntries(
      METHODS.map((method) => [method, evaluateRows(runs[method].predictions)]),
    );
    perTrack = [1, 2, 3].map((id) => ({
      id,
      metrics: Object.fromEntries(
        METHODS.map((method) => [
          method,
          evaluateRows(runs[method].predictions, id),
        ]),
      ),
    }));
    timeline = Array.from({ length: state.horizon }, (_, i) => {
      const frame = i + 1;
      const labels = rowsFor(data.labels).filter(
        (row) => row.frame_id === frame,
      );
      const values = Object.fromEntries(
        METHODS.map((method) => [
          method,
          engine.evaluate(
            rowsFor(runs[method].predictions).filter(
              (row) => row.frame_id === frame,
            ),
            labels,
          ).coordinate_rmse_yards,
        ]),
      );
      return { frame, seconds: frame / 10, ...values };
    });
  }
  function renderMetrics() {
    const metric = selectedMetrics[state.method],
      reference = selectedMetrics.constant_velocity;
    $("rmse").textContent = yards(metric.coordinate_rmse_yards);
    $("cv-rmse").textContent = yards(reference.coordinate_rmse_yards);
    $("hold-rmse").textContent = yards(
      selectedMetrics.hold_last_position.coordinate_rmse_yards,
    );
    const delta =
      metric.coordinate_rmse_yards - reference.coordinate_rmse_yards;
    $("delta").textContent =
      `Selected − velocity: ${delta >= 0 ? "+" : "−"}${Math.abs(delta).toFixed(4)} yd`;
    $("metric-scope").textContent =
      `${metric.entities} track${metric.entities === 1 ? "" : "s"} × ${state.horizon} future frames · ${metric.coordinate_denominator} coordinate errors`;
    $("all-rmse").textContent = yards(
      runs[state.method].metrics.coordinate_rmse_yards,
    );
    $("all-scope").textContent =
      `12 tracks · ${runs[state.method].metrics.rows} forecast rows · fixed game split`;
    $("ade").textContent = yards(metric.ade_yards);
    $("fde").textContent = yards(metric.fde_yards);
    $("track-table").replaceChildren(
      ...perTrack.map((track) => {
        const row = node("tr"),
          cell = node("td"),
          button = node(
            "button",
            `track-button${Number(state.player) === track.id ? " selected" : ""}`,
            trackName(track.id),
          );
        button.id = `track-${track.id}`;
        button.setAttribute(
          "aria-pressed",
          String(Number(state.player) === track.id),
        );
        button.addEventListener("click", () => {
          $("player-select").value = String(track.id);
          state.player = String(track.id);
          refresh();
        });
        cell.append(button);
        row.append(cell);
        for (const method of [
          state.method,
          "constant_velocity",
          "hold_last_position",
        ])
          row.append(
            node(
              "td",
              "",
              track.metrics[method].coordinate_rmse_yards.toFixed(4),
            ),
          );
        return row;
      }),
    );
  }
  function renderTimeline() {
    if (!runs) return;
    const chart = $("timeline"),
      type = fontSize(chart, 620),
      left = Math.max(54, type * 3.8),
      top = 20;
    const box = {
      left,
      top,
      width: 620 - left - 18,
      height: 230 - top - type * 3.7,
    };
    const maximum =
      Math.max(
        0.001,
        ...timeline.flatMap((row) => METHODS.map((method) => row[method])),
      ) * 1.08;
    const x = (frame) =>
      box.left +
      (state.horizon === 1 ? 0.5 : (frame - 1) / (state.horizon - 1)) *
        box.width;
    const y = (error) => box.top + box.height - (error / maximum) * box.height;
    chart.replaceChildren();
    for (let i = 0; i <= 3; i++) {
      const value = (maximum * i) / 3,
        position = y(value);
      chart.append(
        svg("line", {
          x1: box.left,
          y1: position,
          x2: box.left + box.width,
          y2: position,
          stroke: "#dcded3",
        }),
      );
      chart.append(
        svg(
          "text",
          {
            x: box.left - 8,
            y: position + type * 0.3,
            fill: "#59685e",
            "font-size": type,
            "text-anchor": "end",
            "font-family": "ui-monospace,monospace",
          },
          value.toFixed(2),
        ),
      );
    }
    const ticks = [
      ...new Set([1, Math.ceil(state.horizon / 2), state.horizon]),
    ];
    for (const frame of ticks)
      chart.append(
        svg(
          "text",
          {
            x: x(frame),
            y: box.top + box.height + type * 1.6,
            fill: "#59685e",
            "font-size": type,
            "text-anchor": "middle",
            "font-family": "ui-monospace,monospace",
          },
          `${(frame / 10).toFixed(1)}s`,
        ),
      );
    for (const method of [
      "hold_last_position",
      "constant_velocity",
      state.method,
    ]) {
      const selected = method === state.method;
      const color = selected ? COLORS.selected : COLORS[method];
      chart.append(
        svg("polyline", {
          points: timeline
            .map((row) => `${x(row.frame)},${y(row[method])}`)
            .join(" "),
          fill: "none",
          stroke: color,
          "stroke-width": selected ? 2.8 : 2,
          "stroke-dasharray": selected
            ? "none"
            : method === "constant_velocity"
              ? "5 4"
              : "2 4",
          "stroke-linejoin": "round",
        }),
      );
      for (const row of timeline)
        chart.append(
          svg("circle", {
            cx: x(row.frame),
            cy: y(row[method]),
            r: selected ? 2.8 : 1.7,
            fill: color,
          }),
        );
    }
    if (state.frame > 0)
      chart.append(
        svg("line", {
          x1: x(state.frame),
          y1: box.top,
          x2: x(state.frame),
          y2: box.top + box.height,
          stroke: "#20382c",
          "stroke-width": 1,
          opacity: 0.55,
        }),
      );
    chart.setAttribute(
      "aria-label",
      `${state.horizon} future-frame errors for ${state.player === "all" ? "three tracks" : trackName(state.player)}. Selected rule ends at ${timeline.at(-1)[state.method].toFixed(4)} yards coordinate RMSE per frame. Scores in the headline aggregate all requested frames.`,
    );
  }
  function stopAnimation() {
    playing = false;
    if (animation !== null) root.cancelAnimationFrame(animation);
    animation = null;
    startTime = null;
  }
  function renderMoment() {
    updateControls();
    renderField();
    renderTimeline();
  }
  function tick(timestamp) {
    animation = null;
    if (!playing) return;
    if (startTime === null) startTime = timestamp;
    state.frame = Math.min(
      state.horizon,
      startFrame + Math.floor((timestamp - startTime) / 100),
    );
    if (state.frame >= state.horizon) playing = false;
    renderMoment();
    if (playing) animation = root.requestAnimationFrame(tick);
  }
  function refresh() {
    stopAnimation();
    try {
      runs = Object.fromEntries(
        METHODS.map((method) => [
          method,
          engine.run(data, {
            method,
            horizon: state.horizon,
            damping: state.damping,
          }),
        ]),
      );
      if (!fullVelocity)
        fullVelocity = engine.run(data, {
          method: "constant_velocity",
          horizon: 12,
          damping: 1,
        });
      state.frame = Math.min(state.frame, state.horizon);
      calculateMetrics();
      renderMetrics();
      renderMoment();
      $("error").hidden = true;
      $("export").disabled = false;
      $("status").textContent =
        `Computed ${runs[state.method].metrics.rows} holdout forecast rows. No fitting or network access.`;
    } catch (error) {
      runs = null;
      selectedMetrics = null;
      perTrack = [];
      timeline = [];
      $("error").textContent = `Cannot evaluate: ${error.message}`;
      $("error").hidden = false;
      $("status").textContent =
        "No valid result is available. Correct the controls or reset the view.";
      for (const id of [
        "rmse",
        "cv-rmse",
        "hold-rmse",
        "all-rmse",
        "ade",
        "fde",
      ])
        $(id).textContent = "—";
      for (const id of ["metric-scope", "all-scope", "delta"])
        $(id).textContent = "";
      $("field").replaceChildren();
      $("timeline").replaceChildren();
      $("track-table").replaceChildren();
      $("export").disabled = true;
      updateControls();
    }
  }
  function syncDefaults() {
    $("play-select").value = "105:1";
    $("player-select").value = "all";
    $("method").value = "damped_velocity";
    $("horizon").value = "12";
    $("damping").value = "0.9";
    $("truth").checked = false;
    $("view").value = "routes";
    Object.assign(state, {
      game: 105,
      play: 1,
      player: "all",
      method: "damped_velocity",
      horizon: 12,
      damping: 0.9,
      frame: 12,
      truth: false,
      view: "routes",
    });
  }
  $("play-select").replaceChildren(
    ...[105, 106].flatMap((game) =>
      [1, 2].map((play) => {
        const option = node(
          "option",
          "",
          `Game ${game} · Play ${String(play).padStart(2, "0")}`,
        );
        option.value = `${game}:${play}`;
        return option;
      }),
    ),
  );
  $("play-select").addEventListener("change", () => {
    [state.game, state.play] = $("play-select").value.split(":").map(Number);
    refresh();
  });
  $("player-select").addEventListener("change", () => {
    state.player = $("player-select").value;
    refresh();
  });
  $("method").addEventListener("change", () => {
    state.method = $("method").value;
    refresh();
  });
  $("horizon").addEventListener("input", () => {
    state.horizon = Number($("horizon").value);
    refresh();
  });
  $("damping").addEventListener("input", () => {
    state.damping = Number($("damping").value);
    refresh();
  });
  $("truth").addEventListener("change", () => {
    state.truth = $("truth").checked;
    renderMoment();
  });
  $("view").addEventListener("change", () => {
    state.view = $("view").value;
    renderField();
  });
  $("frame").addEventListener("input", () => {
    stopAnimation();
    state.frame = Math.max(
      0,
      Math.min(state.horizon, Number($("frame").value)),
    );
    renderMoment();
  });
  $("previous").addEventListener("click", () => {
    stopAnimation();
    state.frame = Math.max(0, state.frame - 1);
    renderMoment();
  });
  $("next").addEventListener("click", () => {
    stopAnimation();
    state.frame = Math.min(state.horizon, state.frame + 1);
    renderMoment();
  });
  $("play").addEventListener("click", () => {
    if (playing) {
      stopAnimation();
      updateControls();
      return;
    }
    if (!runs || reducedMotion) return;
    if (state.frame >= state.horizon) state.frame = 0;
    playing = true;
    startFrame = state.frame;
    startTime = null;
    renderMoment();
    animation = root.requestAnimationFrame(tick);
  });
  $("reset").addEventListener("click", () => {
    syncDefaults();
    refresh();
  });
  $("export").addEventListener("click", () => {
    if (!runs) return;
    const report = {
      schema_version: 1,
      evidence_type: "SYNTHETIC_ONLY",
      purpose:
        "Local demonstration of transparent motion rules, not a learned model or competition evaluation.",
      selection: { ...state },
      units: "yards",
      sampling_hz: 10,
      observed_frames: 8,
      provenance: data.provenance,
      observations: data.observations.filter(samePlay),
      requests: data.requests.filter(
        (row) => chosen(row) && row.frame_id <= state.horizon,
      ),
      predictions: rowsFor(runs[state.method].predictions),
      evaluation_labels: rowsFor(data.labels),
      selected_metrics: selectedMetrics,
      per_track: perTrack,
      error_timeline: timeline,
      holdout_metrics: Object.fromEntries(
        METHODS.map((method) => [method, runs[method].metrics]),
      ),
      checks: runs[state.method].checks,
      limitations: runs[state.method].limitations,
    };
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }),
    );
    const link = node("a");
    link.href = url;
    link.download = `synthetic-route-game${state.game}-play${state.play}.json`;
    link.click();
    root.setTimeout(() => URL.revokeObjectURL(url), 1000);
    $("status").textContent =
      "Downloaded synthetic inputs, predictions, separate evaluation labels and exact metrics.";
  });
  root.addEventListener("resize", () => {
    if (runs) renderMoment();
  });
  syncDefaults();
  refresh();
})(typeof globalThis !== "undefined" ? globalThis : window);
