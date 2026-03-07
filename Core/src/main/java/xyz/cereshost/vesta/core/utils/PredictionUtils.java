package xyz.cereshost.vesta.core.utils;

import lombok.experimental.UtilityClass;

@UtilityClass
public class PredictionUtils {

    public static float clampRatio(float ratio) {
        if (!Float.isFinite(ratio)) return 0f;
        if (ratio > 1f) return 1f;
        if (ratio < -1f) return -1f;
        return ratio;
    }

    /**
     * Convierte (totalMove, ratio) a (upMove, downMove).
     * ratio en [-1,1], positivo indica dominio del maximo.
     */
    public static float[] splitMoves(float totalMove, float ratio) {
        float total = Math.max(0f, totalMove);
        float r = clampRatio(ratio);
        if (total <= 0f) return new float[]{0f, 0f};

        float upMove;
        float downMove;
        if (r >= 0f) {
            upMove = total / (2f - r);
            downMove = total - upMove;
        } else {
            downMove = total / (2f + r);
            upMove = total - downMove;
        }
        return new float[]{Math.max(0f, upMove), Math.max(0f, downMove)};
    }

    /**
     * Direccion por signo sin umbral.
     * ratio positivo => LONG, ratio negativo => SHORT.
     */
    public static int directionFromRatioRaw(float ratio) {
        if (!Float.isFinite(ratio)) return 0;
        if (ratio > 0f) return 1;
        if (ratio < 0f) return -1;
        return 0;
    }

    /**
     * Direccion usando umbral.
     */
    public static int directionFromRatioThreshold(float ratio, float threshold) {
        if (!Float.isFinite(ratio)) return 0;
        if (ratio > threshold) return 1;
        if (ratio < -threshold) return -1;
        return 0;
    }
}
