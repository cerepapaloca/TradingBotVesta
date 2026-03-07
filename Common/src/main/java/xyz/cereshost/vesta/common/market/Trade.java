package xyz.cereshost.vesta.common.market;

public record Trade(long time, float price, float qty, boolean isBuyerMaker) {

    public double quoteQty(){
        return price * qty;
    }

    @Override
    public int hashCode() {
        return (int) time;
    }

}
