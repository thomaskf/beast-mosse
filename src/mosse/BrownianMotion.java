package mosse;

import beast.base.inference.Distribution;
import beast.base.inference.State;

import java.util.List;
import java.util.Random;

public class BrownianMotion extends Distribution {
    @Override
    public List<String> getArguments() {
        return null;
    }

    @Override
    public List<String> getConditions() {
        return null;
    }

    @Override
    public void sample(State state, Random random) {

    }
}
