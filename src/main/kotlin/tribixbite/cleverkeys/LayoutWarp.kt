package tribixbite.cleverkeys

import android.graphics.PointF

/**
 * Per-key layout warp for swipe typing (separable piecewise-linear).
 *
 * Maps raw touch coordinates from the user's ACTUAL key layout into the model's
 * canonical QWERTY normalized [0,1] space, so the QWERTY-trained ONNX swipe model
 * works on offset / scaled / per-row-shifted layouts.
 *
 * WHY THIS EXISTS
 * ---------------
 * `SwipeTrajectoryProcessor.normalizeCoordinates()` does a single global affine
 * (`nx = rawX / keyboardWidth`) and then `detectNearestKeys()` snaps to the fixed
 * canonical `KeyboardGrid`. A global affine cannot correct per-row / per-key
 * offsets, so custom layouts mis-decode. Measured on "allison's QWERTY" (blank
 * spacers left of q/a, added ';'/'.' columns -> keyboard is 11 key-units wide, not
 * 10): the bottom row compresses left and  b -> 'v', n -> 'b', m -> 'n'.
 *
 * HOW IT WORKS  (separable: X and Y handled independently)
 * -------------------------------------------------------
 *   Y:  piecewise-linear through the 3 real row-center Ys -> canonical row centers
 *       (1/6, 1/2, 5/6), extrapolated past the ends.
 *   X:  each canonical row gets its own piecewise-linear map built from THAT row's
 *       key centers (real pixel X -> canonical X), extrapolated past the ends. For a
 *       point we evaluate the two bracketing rows' X-maps and blend them by the
 *       point's fractional row position (the same fraction used for Y), so there is
 *       no discontinuity crossing rows -> clean velocity/acceleration features.
 *
 * Result: EXACT at every key (a touch on key c -> canonical center of c), linear in
 * between, and sensibly extrapolated at the edges (no IDW convex-hull flattening).
 *
 * SCOPE: this is a separable deformation warp — it corrects horizontal offset/scale
 * (per row) and vertical offset/scale. That covers offset/scaled/shifted QWERTY
 * (e.g. this layout). It assumes the physical rows still hold QWERTY's row
 * assignment (q-row on top, a-row middle, z-row bottom). True key *rearrangements*
 * (AZERTY: a/q/w/z/m swapped between rows) are NOT separable and need a 2D scattered
 * warp (triangulation/TPS) instead — out of scope here.
 *
 * STATUS: prototype. Needs on-device testing and a yOffset check (see `apply`).
 */
object LayoutWarp {

    private val ROWS = arrayOf("qwertyuiop", "asdfghjkl", "zxcvbnm")
    private val CANON_ROW_Y = floatArrayOf(1f / 6f, 0.5f, 5f / 6f)

    /**
     * Build a reusable warp from the current real key positions (pixels). Returns
     * null if there aren't enough anchors or the rows aren't vertically ordered —
     * caller should then fall back to the existing affine normalization.
     * Build ONCE per swipe and reuse `apply` for every point.
     */
    fun build(realKeys: Map<Char, PointF>?): Warp? {
        if (realKeys.isNullOrEmpty()) return null

        val realRowY = FloatArray(3)
        val realXs = arrayOfNulls<FloatArray>(3)
        val canonXs = arrayOfNulls<FloatArray>(3)

        for (r in 0..2) {
            // (realX, canonX) for every letter of this canonical row that has a position
            val pairs = ArrayList<Pair<Float, Float>>(10)
            var ySum = 0f
            var yN = 0
            for (ch in ROWS[r]) {
                val real = realKeys[ch] ?: continue
                val canon = KeyboardGrid.getKeyPosition(ch) ?: continue
                pairs.add(real.x to canon.x)
                ySum += real.y
                yN++
            }
            if (pairs.size < 2) return null            // need a slope per row
            pairs.sortBy { it.first }                  // ascending real X
            // reject duplicate/again non-monotonic X (degenerate)
            for (i in 1 until pairs.size) if (pairs[i].first <= pairs[i - 1].first) return null
            realXs[r] = FloatArray(pairs.size) { pairs[it].first }
            canonXs[r] = FloatArray(pairs.size) { pairs[it].second }
            realRowY[r] = ySum / yN
        }
        // rows must be top->bottom (Y increases downward)
        if (!(realRowY[0] < realRowY[1] && realRowY[1] < realRowY[2])) return null

        @Suppress("UNCHECKED_CAST")
        return Warp(realRowY, realXs as Array<FloatArray>, canonXs as Array<FloatArray>)
    }

    class Warp internal constructor(
        private val realRowY: FloatArray,    // size 3, ascending
        private val realXs: Array<FloatArray>,
        private val canonXs: Array<FloatArray>
    ) {
        /**
         * Warp one raw touch point to canonical normalized [0,1].
         *
         * @param yOffset finger-occlusion compensation (px), ADDED to py before
         *   matching (mirrors SwipeTrajectoryProcessor.touchYOffset). If results land
         *   one row off, flip the sign here. Pass the processor's touchYOffset or 0.
         */
        fun apply(px: Float, py: Float, yOffset: Float = 0f): PointF {
            val ay = py + yOffset

            // Y: piecewise-linear realRowY -> canonical row centers
            val ny = pwl(ay, realRowY, CANON_ROW_Y)

            // fractional row position of ay among the 3 row centers
            val rf: Float = when {
                ay <= realRowY[0] -> 0f
                ay >= realRowY[2] -> 2f
                ay < realRowY[1] -> (ay - realRowY[0]) / (realRowY[1] - realRowY[0])          // 0..1
                else -> 1f + (ay - realRowY[1]) / (realRowY[2] - realRowY[1])                 // 1..2
            }
            val r0 = rf.toInt().coerceIn(0, 1)   // bracketing rows r0, r0+1
            val frac = (rf - r0).coerceIn(0f, 1f)

            // X: evaluate both bracketing rows' maps, blend by row fraction
            val x0 = pwl(px, realXs[r0], canonXs[r0])
            val x1 = pwl(px, realXs[r0 + 1], canonXs[r0 + 1])
            val nx = x0 + frac * (x1 - x0)

            return PointF(nx.coerceIn(0f, 1f), ny.coerceIn(0f, 1f))
        }

        /** Piecewise-linear map of `v` through control points (xs -> ys); xs ascending.
         *  Linear extrapolation beyond the ends. */
        private fun pwl(v: Float, xs: FloatArray, ys: FloatArray): Float {
            val n = xs.size
            if (n == 1) return ys[0]
            if (v <= xs[0]) {
                val s = (ys[1] - ys[0]) / (xs[1] - xs[0])
                return ys[0] + (v - xs[0]) * s
            }
            if (v >= xs[n - 1]) {
                val s = (ys[n - 1] - ys[n - 2]) / (xs[n - 1] - xs[n - 2])
                return ys[n - 1] + (v - xs[n - 1]) * s
            }
            var i = 0
            while (i < n - 1 && v > xs[i + 1]) i++
            val t = (v - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
        }
    }
}
