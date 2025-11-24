# Testing the Mushroom Classifier Streamlit App

## How to Run
1. Open your browser and go to: **http://localhost:8501**
2. If the app isn't running, execute: `py -m streamlit run app.py`

## Sample Test Cases

### Test Case 1: Edible Mushroom (should be edible)
Based on the rules: odor=almond/anise/none AND spore-print-color=NOT green

- **cap-shape**: x (convex)
- **cap-surface**: s (smooth)
- **cap-color**: w (white)
- **bruises**: t (bruises)
- **odor**: a (almond) ← edible indicator
- **gill-attachment**: f (free)
- **gill-spacing**: c (close)
- **gill-size**: b (broad)
- **gill-color**: w (white)
- **stalk-shape**: e (enlarging)
- **stalk-root**: e (equal)
- **stalk-surface-above-ring**: s (smooth)
- **stalk-surface-below-ring**: s (smooth)
- **stalk-color-above-ring**: w (white)
- **stalk-color-below-ring**: w (white)
- **veil-color**: w (white)
- **ring-number**: o (one)
- **ring-type**: p (pendant)
- **spore-print-color**: w (white) ← not green
- **population**: s (scattered)
- **habitat**: g (grasses)

### Test Case 2: Poisonous Mushroom (should be poisonous)
Based on the rules: odor=NOT(almond/anise/none)

- **cap-shape**: x (convex)
- **cap-surface**: s (smooth)
- **cap-color**: n (brown)
- **bruises**: t (bruises)
- **odor**: f (foul) ← poisonous indicator
- **gill-attachment**: f (free)
- **gill-spacing**: c (close)
- **gill-size**: b (broad)
- **gill-color**: n (brown)
- **stalk-shape**: e (enlarging)
- **stalk-root**: e (equal)
- **stalk-surface-above-ring**: s (smooth)
- **stalk-surface-below-ring**: s (smooth)
- **stalk-color-above-ring**: w (white)
- **stalk-color-below-ring**: w (white)
- **veil-color**: w (white)
- **ring-number**: o (one)
- **ring-type**: p (pendant)
- **spore-print-color**: k (black)
- **population**: a (abundant)
- **habitat**: u (urban)

## Attribute Value Codes

- **cap-shape**: b, c, x, f, k, s
- **cap-surface**: f, g, y, s
- **cap-color**: n, b, c, g, r, p, u, e, w, y
- **bruises**: t, f
- **odor**: a, l, c, y, f, m, n, p, s
- **gill-attachment**: a, d, f, n
- **gill-spacing**: c, w, d
- **gill-size**: b, n
- **gill-color**: k, n, b, h, g, r, o, p, u, e, w, y
- **stalk-shape**: e, t
- **stalk-root**: b, c, u, e, z, r, ? (missing)
- **stalk-surface-above-ring**: f, y, k, s
- **stalk-surface-below-ring**: f, y, k, s
- **stalk-color-above-ring**: n, b, c, g, o, p, e, w, y
- **stalk-color-below-ring**: n, b, c, g, o, p, e, w, y
- **veil-color**: n, o, w, y
- **ring-number**: n, o, t
- **ring-type**: c, e, f, l, n, p, s, z
- **spore-print-color**: k, n, b, h, r, o, u, w, y
- **population**: a, c, n, s, v, y
- **habitat**: g, l, m, p, u, w, d

