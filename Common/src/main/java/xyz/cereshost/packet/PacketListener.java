package xyz.cereshost.packet;

import com.google.gson.reflect.TypeToken;
import lombok.Getter;

import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;

@SuppressWarnings("unchecked")
@Getter
public abstract class PacketListener<T extends Packet> {

    private final Class<T> clazz;

    public PacketListener(){
        Type type = ((ParameterizedType) getClass().getGenericSuperclass()).getActualTypeArguments()[0];
        this.clazz = (Class<T>) TypeToken.get(type).getRawType();
        BasePacketHandler.addListener(this);
    }

    public T decodePacketAndReceive(byte[] data) {
        T packet = (T) PacketManager.decodePacket(data);
        try {
            onReceive(packet);
        }catch (Exception e){
            e.printStackTrace();
        }
        return packet;
    }

    public abstract void onReceive(T packet);
}
