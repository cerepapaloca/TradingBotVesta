package xyz.cereshost.utils;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterServer;
import ai.djl.training.ParameterStore;
import lombok.SneakyThrows;
import xyz.cereshost.engine.VestaEngine;

import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;

public class VestaParameterStore extends ParameterStore {

    public VestaParameterStore(NDManager manager, boolean store) {
        super(manager, store);
    }

    private static final int numThreads = 8;

    @SuppressWarnings("unchecked")
    @SneakyThrows
    @Override
    public void updateAllParameters() {
        Field fieldManager = ParameterStore.class.getDeclaredField("parameterMap");
        fieldManager.setAccessible(true);

        Class<?> clazzParameterData = Class.forName("ai.djl.training.ParameterStore$ParameterData");

        Field parameterField = clazzParameterData.getDeclaredField("parameter");
        parameterField.setAccessible(true);
        Field parameterServerField = ParameterStore.class.getDeclaredField("parameterServer");
        parameterServerField.setAccessible(true);
        Method toArray = clazzParameterData.getDeclaredMethod("toArray");
        toArray.setAccessible(true);

        ParameterServer parameterServer = (ParameterServer) parameterServerField.get((ParameterStore) this);
        Map<String, ?> parameterStore = (Map<String, ?>) fieldManager.get((ParameterStore) this);

        List<Map.Entry<String, ?>> list = new ArrayList(parameterStore.entrySet());
        int size = list.size();
        int chunkSize = (size + numThreads - 1) / numThreads;
        CountDownLatch latch = new CountDownLatch(numThreads);

        for (int i = 0; i < numThreads; i++) {
            int start = i * chunkSize;
            int end = Math.min(start + chunkSize, size);

            if (start >= end) {
                latch.countDown();
                continue;
            }

            List<Map.Entry<String, ?>> subList = list.subList(start, end);

            VestaEngine.EXECUTOR_TRAINING.submit(() -> {
                try {
                    for (Map.Entry<String, ?> entry : subList) {
                        String parameterId = entry.getKey();
                        Object data = entry.getValue();


                        Parameter parameter = (Parameter) parameterField.get(data);
                        if (parameter.requiresGradient()) {
                            NDArray[] params = (NDArray[]) toArray.invoke(data);
                            parameterServer.update(parameterId, params);
                        }
                    }
                } catch (IllegalAccessException | InvocationTargetException ignored) {
                } finally {
                    latch.countDown();
                }
            });
        }

        latch.await();
    }
}
