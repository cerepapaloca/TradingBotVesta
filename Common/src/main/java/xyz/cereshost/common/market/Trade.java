package xyz.cereshost.common.market;

public record Trade(long id, long time, float price, float qty, boolean isBuyerMaker) {

    public double quoteQty(){
        return price * qty;
    }

    @Override
    public int hashCode() {
        return (int) id;
    }

}
