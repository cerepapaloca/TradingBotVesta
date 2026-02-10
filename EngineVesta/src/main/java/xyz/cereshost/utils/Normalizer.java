package xyz.cereshost.utils;

public interface Normalizer<T> {

    public void fit(T source);

    T transform (T source);
    T inverseTransform (T source);
}
