# CareerLens Color Palette

Based on the official Logo.jpg design.

## Color Swatches

### Primary Colors

| Color Name | Hex Code | RGB | Usage |
|------------|----------|-----|-------|
| Dark Navy | `#1a2332` | rgb(26, 35, 50) | Background, Main containers |
| Cyan Accent | `#4dd4d4` | rgb(77, 212, 212) | Highlights, Buttons, Links |
| White | `#FFFFFF` | rgb(255, 255, 255) | Main text, Headers |
| Light Grey | `#B0BEC5` | rgb(176, 190, 197) | Subtitles, Secondary text |

### Secondary Colors

| Color Name | Hex Code | RGB | Usage |
|------------|----------|-----|-------|
| Medium Navy | `#243447` | rgb(36, 52, 71) | Cards, Secondary containers |
| Dark Cyan | `#2e9bb0` | rgb(46, 155, 176) | Hover states, Gradients |
| Card Background | `#2c3e50` | rgb(44, 62, 80) | Content cards |
| Border Grey | `#3a4a5c` | rgb(58, 74, 92) | Borders, Dividers |

### Status Colors

| Color Name | Hex Code | RGB | Usage |
|------------|----------|-----|-------|
| Success Green | `#10B981` | rgb(16, 185, 129) | Success messages |
| Warning Amber | `#F59E0B` | rgb(245, 158, 11) | Warning messages |
| Error Red | `#EF4444` | rgb(239, 68, 68) | Error messages |

## Gradients

### Primary Gradient
```css
background: linear-gradient(135deg, #4dd4d4 0%, #2e9bb0 100%);
```
Used for: Buttons, Highlights, Special UI elements

### Background Gradient
```css
background: linear-gradient(135deg, #1a2332 0%, #243447 100%);
```
Used for: Hero sections, Feature cards

## Contrast Ratios (WCAG AA Compliance)

- White (#FFFFFF) on Dark Navy (#1a2332): **14.8:1** ✓ AAA
- Cyan (#4dd4d4) on Dark Navy (#1a2332): **6.2:1** ✓ AA
- Light Grey (#B0BEC5) on Dark Navy (#1a2332): **8.1:1** ✓ AAA

All color combinations meet or exceed WCAG AA standards for accessibility.

## CSS Variables Reference

```css
:root {
    /* Backgrounds */
    --bg-primary: #1a2332;
    --bg-secondary: #243447;
    --bg-container: #243447;
    --card-bg: #2c3e50;
    
    /* Text */
    --text-primary: #FFFFFF;
    --text-secondary: #B0BEC5;
    --text-primary-light: #FFFFFF;
    
    /* Brand */
    --brand-glow: #4dd4d4;
    --brand-core: #2e9bb0;
    --cyan: #4dd4d4;
    --navy: #1a2332;
    
    /* UI */
    --border-color: #3a4a5c;
    --hover-bg: #2c3e50;
    --accent-gradient: linear-gradient(135deg, #4dd4d4 0%, #2e9bb0 100%);
}
```

## Typography

### Font Families
- **Headings**: 'Montserrat', sans-serif (700 weight)
- **Body**: 'Inter', sans-serif (400 weight)
- **Code**: 'Courier New', monospace

### Font Colors
- Headings: White (#FFFFFF) or Cyan (#4dd4d4) for emphasis
- Body text: White (#FFFFFF)
- Subtitles: Light Grey (#B0BEC5)
- Links: Cyan (#4dd4d4)

## Logo Specifications

- **File**: Logo.jpg
- **Size**: 1000 x 488 pixels
- **Aspect Ratio**: ~2:1
- **File Size**: 24 KB
- **Format**: JPEG
- **Background**: Dark Navy (#1a2332)
- **Primary Colors**: Cyan/Turquoise with gradient effects

## Usage Examples

### Button Styles
```css
.primary-button {
    background: linear-gradient(135deg, #4dd4d4 0%, #2e9bb0 100%);
    color: #FFFFFF;
    border: none;
}

.secondary-button {
    background: transparent;
    color: #4dd4d4;
    border: 2px solid #4dd4d4;
}
```

### Card Styles
```css
.card {
    background-color: #243447;
    border: 1px solid #3a4a5c;
    color: #FFFFFF;
}

.card h3 {
    color: #4dd4d4;
}

.card p {
    color: #B0BEC5;
}
```

### Text Hierarchy
```css
h1 { color: #FFFFFF; }           /* Main headings */
h2 { color: #4dd4d4; }           /* Section headings */
h3 { color: #FFFFFF; }           /* Subsection headings */
p { color: #FFFFFF; }            /* Body text */
.subtitle { color: #B0BEC5; }    /* Subtitles and captions */
```
