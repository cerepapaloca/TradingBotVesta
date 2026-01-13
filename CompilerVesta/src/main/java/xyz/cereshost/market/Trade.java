package xyz.cereshost.market;

public record Trade(long id, long time, double price, double qty, boolean isBuyerMaker) {

    public double quoteQty(){
        return price * qty;
    }

    @Override
    public int hashCode() {
        return (int) id;
    }

}
