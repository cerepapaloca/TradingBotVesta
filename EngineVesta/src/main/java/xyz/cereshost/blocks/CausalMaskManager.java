package xyz.cereshost.blocks;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import org.jetbrains.annotations.NotNull;

import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * CausalMaskManager:
 * - crea las máscaras en un subManager dedicado (uno por key device:T:dtype)
 * - expone getMaskAndDerived(rootMgr, T, dtype) -> {mask, invMask, negInf}
 * - soporta update y clearSubManagers() para cerrar subManagers y liberar RAM
 *
 * Uso recomendado:
 * - Al inicio del forward: NDArray[] triple = CausalMaskManager.getMaskAndDerived(rootMgr, T, dtype);
 * - Si necesitas la máscara en tu subManager local: maskCopy = triple[0].toDevice(localSubMgr.getDevice(), false);
 * - Al final de cada epoch (desde hilo principal cuando no haya forwards en vuelo):
 *      CausalMaskManager.clearSubManagers();
 */
public final class CausalMaskManager {

    private static final ConcurrentHashMap<String, MaskHolder> CACHE = new ConcurrentHashMap<>();

    private CausalMaskManager() {}

    private static String key(Device device, int T, DataType dtype) {
        return device.toString() + ":" + T + ":" + dtype.toString();
    }

    private static final class MaskHolder {
        final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
        // Aquí guardamos el subManager y los NDArrays creados en dicho subManager
        volatile NDManager subManager;    // creado con rootMgr.newSubManager()
        volatile NDArray mask;            // [1,1,T,T] float en subManager
        volatile NDArray invMask;         // 1 - mask en subManager
        volatile NDArray negInf;          // scalar -1e9 en subManager
        final Device device;
        final int T;
        final DataType dtype;

        MaskHolder(Device device, int T, DataType dtype) {
            this.device = device;
            this.T = T;
            this.dtype = dtype;
        }

        void closeAllAndSubManager() {
            try { if (mask != null) mask.close(); } catch (Exception ignored) {}
            try { if (invMask != null) invMask.close(); } catch (Exception ignored) {}
            try { if (negInf != null) negInf.close(); } catch (Exception ignored) {}
            try { if (subManager != null) subManager.close(); } catch (Exception ignored) {}
            mask = null;
            invMask = null;
            negInf = null;
            subManager = null;
        }
    }

    private static MaskHolder getOrCreateHolder(Device device, int T, DataType dtype) {
        String k = key(device, T, dtype);
        return CACHE.computeIfAbsent(k, kk -> new MaskHolder(device, T, dtype));
    }

    private static int idxForRemoveCache = 1;

    /**
     * Devuelve {mask, invMask, negInf}. Si no existen, los crea **en un subManager** (subManager = rootMgr.newSubManager()).
     * Los NDArrays están creados en dicho subManager.
     *
     * Nota: No cerrar los NDArrays devueltos aquí (están gestionados por el manager interno).
     */
    public static NDArray[] getMaskAndDerived(@NotNull NDManager rootMgr, int T, DataType dtype, int oftenCache) {
        Objects.requireNonNull(rootMgr, "root manager");
        if ((idxForRemoveCache % oftenCache) == 0) clearSubManagers();
        idxForRemoveCache++;

        Device device = rootMgr.getDevice();
        MaskHolder holder = getOrCreateHolder(device, T, dtype);
        // Fast read path
        holder.lock.readLock().lock();
        try {
            if (holder.subManager != null && holder.mask != null && holder.invMask != null && holder.negInf != null) {
                return new NDArray[]{holder.mask, holder.invMask, holder.negInf};
            }
        } finally {
            holder.lock.readLock().unlock();
        }

        // Need to create -> take write lock
        holder.lock.writeLock().lock();
        try {
            // double-check
            if (holder.subManager == null || holder.mask == null || holder.invMask == null || holder.negInf == null) {
                // Create a subManager using the provided root manager to ensure correct device/context
                // If holder already had a subManager but arrays missing, reuse it; otherwise create new
                if (holder.subManager == null) {
                    holder.subManager = rootMgr.newSubManager();
                }
                NDManager mgr = holder.subManager;

                // create mask in the subManager to avoid populating root manager memory
                NDArray idx = mgr.arange(T);
                NDArray iIdx = idx.expandDims(0);
                NDArray jIdx = idx.expandDims(1);
                NDArray mask2d = jIdx.lte(iIdx); // [T,T] boolean
                // expand to [1,1,T,T] and convert dtype
                NDArray mask4d = mask2d.expandDims(0).expandDims(0).toType(dtype, false);

                // derived arrays
                NDArray inv = mask4d.mul(-1f).add(1f);
                NDArray neg = mgr.full(new Shape(1), -1e9f).toType(dtype, false);

                // if there were old arrays, close them (safe because we hold write lock)
                try {
                    if (holder.mask != null) holder.mask.close();
                    if (holder.invMask != null) holder.invMask.close();
                    if (holder.negInf != null) holder.negInf.close();
                } catch (Exception ignored) {}

                holder.mask = mask4d;
                holder.invMask = inv;
                holder.negInf = neg;
            }
            return new NDArray[]{holder.mask, holder.invMask, holder.negInf};
        } finally {
            holder.lock.writeLock().unlock();
        }
    }

    /**
     * Cierra todos los subManagers y sus NDArrays. Llamar al final de epoch cuando NO haya forwards en vuelo.
     * Esto libera la RAM asociada a las máscaras (las entradas en CACHE permanecen; la próxima llamada recreará
     * subManagers/mascaras según sea necesario).
     */
    public static void clearSubManagers() {
        for (Map.Entry<String, MaskHolder> e : CACHE.entrySet()) {
            MaskHolder h = e.getValue();
            // use write lock to ensure no readers are active
            h.lock.writeLock().lock();
            try {
                if (h.subManager != null) {
                    h.closeAllAndSubManager(); // closes arrays and subManager
                }
            } finally {
                h.lock.writeLock().unlock();
            }
        }
//        System.gc();
    }

    /**
     * Cierra TODO: subManagers y borra la cache completa. Use en shutdown completo.
     */
    public static void clearAll() {
        for (Map.Entry<String, MaskHolder> e : CACHE.entrySet()) {
            MaskHolder h = e.getValue();
            h.lock.writeLock().lock();
            try {
                h.closeAllAndSubManager();
            } finally {
                h.lock.writeLock().unlock();
            }
        }
        CACHE.clear();
    }
}
