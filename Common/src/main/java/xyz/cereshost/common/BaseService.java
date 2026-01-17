package xyz.cereshost.common;

import lombok.Getter;

public abstract class BaseService {

    @Getter
    protected final String name;

    protected BaseService(String name) {
        this.name = name;
    }
}
