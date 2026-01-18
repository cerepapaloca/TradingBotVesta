package xyz.cereshost.common.packet;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.reflections.Reflections;
import lombok.experimental.UtilityClass;
import xyz.cereshost.common.Utils;
import xyz.cereshost.common.Vesta;

import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

@UtilityClass
public class PacketManager {

    private final HashMap<Integer, Class<? extends Packet>> packets = new HashMap<>();

    /**
     * Decodifica el paquete
     * @param data Los datos decodificados
     * @return Paquete decodificado
     */

    public Packet decodePacket(byte[] data) {
        String dataString = new String(data, StandardCharsets.UTF_8);
        JsonObject obj = JsonParser.parseString(dataString).getAsJsonObject();
        int id = obj.get("id").getAsInt();
        // Se usa packets para saber a qué se tiene que decodificar
        return Utils.GSON.fromJson(dataString, packets.get(id));
    }

    /**
     * Codifica un paquete para se enviado
     * @param packet El paquete a codificar
     * @return Los datos códificado
     */

    public byte[] encodePacket(Packet packet) {
        String dataString = Utils.GSON.toJson(packet);
        return dataString.getBytes(StandardCharsets.UTF_8);
    }

    /**
     * A través de los datos sin codificar se obtiene la clase que pertenece
     * @param data El paquete sin codificar
     * @return Clase a la que pertenece
     */

    public Class<? extends Packet> getPacketClass(byte[] data) {
        String dataString = new String(data, StandardCharsets.UTF_8);
        JsonObject obj = JsonParser.parseString(dataString).getAsJsonObject();
        int id = obj.get("id").getAsInt();
        return packets.get(id);
    }

    /**
     * Cual es la ID de la clase
     * @param packet paquete
     * @return ID al que pertenece
     */

    public int whatIsMyId(Class<? extends Packet> packet) {
        for (Map.Entry<Integer, Class<? extends Packet>> entry : packets.entrySet()) {
            if (entry.getValue() == packet) {
                return entry.getKey();
            }
        }
        return -1;
    }

    static {
        Reflections reflections = new Reflections("xyz.cereshost.common.packet");
        // Obtiene todas las clases
        Set<Class<? extends Packet>> packetClasses = reflections.getSubTypesOf(Packet.class);
        int i = 0;
        for (Class<? extends Packet> clazz : packetClasses) {
            packets.put(i, clazz);
            i++;
        }
    }
}
