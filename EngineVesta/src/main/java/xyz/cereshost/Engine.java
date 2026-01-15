package xyz.cereshost;

import ai.djl.Device;
import ai.djl.engine.StandardCapabilities;

public class Engine {

    public static void checkEngines() {
        System.out.println("=== Verificando Engines DJL ===");

        // Listar todos los engines disponibles
        for (String engineName : ai.djl.engine.Engine.getAllEngines()) {
            System.out.println("\nEngine: " + engineName);
            ai.djl.engine.Engine engine = ai.djl.engine.Engine.getEngine(engineName);
            if (engine != null) {
                System.out.println("  Version: " + engine.getVersion());
                System.out.println("  Dispositivos disponibles:");

                for (Device device : engine.getDevices()) {
                    System.out.println("    - " + device +
                            " (GPU: " + device.isGpu() +
                            ", ID: " + device.getDeviceId() +
                            ", C: " + engine.hasCapability(StandardCapabilities.CUDA) + ")");
                }
            } else {
                System.out.println("  No disponible");
            }
        }
    }
}
