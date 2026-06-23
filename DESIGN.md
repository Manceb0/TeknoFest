# QuisMotion Design System

## Direction

A night-time roadside operations console: near-black neutral surfaces, disciplined indigo controls, cyan telemetry, and amber only for elevated network or safety states. Restrained color strategy with dense but legible information.

## Color

- Background: `oklch(0.095 0 0)`
- Surface: `oklch(0.145 0.012 270)`
- Surface raised: `oklch(0.19 0.018 270)`
- Ink: `oklch(0.94 0.01 270)`
- Muted: `oklch(0.69 0.025 270)`
- Primary: `oklch(0.58 0.19 270)`
- Telemetry: `oklch(0.78 0.13 205)`
- QoD: `oklch(0.76 0.16 72)`
- Danger: `oklch(0.65 0.2 25)`

## Typography

Use IBM Plex Sans for interface text and IBM Plex Mono for telemetry and machine-readable values. Compact product scale, tabular numerals, and no decorative display type.

## Components

Controls use 10px radii, clear focus rings, and 160–220ms state transitions. Panels are separated by surface tone or single-pixel low-contrast borders, never broad shadows. Status badges always include a text label.

## Layout

Desktop uses a compact top navigation and a two-column live workspace, with the video occupying the larger column. Mobile collapses to one column and keeps session controls near the video.

## Motion

Motion communicates live state only: connection pulse, risk meter interpolation, and QoD activation. Reduced-motion mode disables continuous animation.
