import sys
from argparse import ArgumentParser
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.append(".")
from spiking.torch.layers.linear import LinearLoihiLIF2
from spiking.torch.utils.surrogates import get_spike_fn
from spiking.torch.utils.quantization import remove_quantization


# matplotlib
plt.rcParams.update({"font.size": 14})
color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# sim params
SEED = 41
STEPS = 100

# global network parameters
IN_SIZE, HID_SIZE, OUT_SIZE = 12, 6, 3
LEAK_I = 1024.0
LEAK_V = 128.0
THRESH = 1024.0


def apply_nonzero(data, axis):
    # make array if not
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    assert data.ndim == 2, "Array must be 2D"
    nonzero = []
    for row in np.rollaxis(data, axis):
        nonzero.append(np.flatnonzero(row))
    return nonzero


class SNN(nn.Module):
    def __init__(self, scale=True):
        super().__init__()

        # parameters, fixed
        leak_i = torch.tensor(LEAK_I / 4096)
        leak_v = torch.tensor(LEAK_V / 4096)
        if scale:
            thresh = torch.tensor(THRESH * 2**6)
        else:
            thresh = torch.tensor((THRESH / 256))

        params = dict(leak_i=leak_i, leak_v=leak_v, thresh=thresh)

        # layers
        self.layer1 = LinearLoihiLIF2(IN_SIZE, HID_SIZE, params, {}, get_spike_fn("BaseSpike"))
        self.layer2 = LinearLoihiLIF2(HID_SIZE, OUT_SIZE, params, {}, get_spike_fn("BaseSpike"))

        # scale weights such that we can compare currents/voltages with Loihi
        if scale:
            remove_quantization(self.layer1.ff, "weight")
            remove_quantization(self.layer2.ff, "weight")
            with torch.no_grad():
                self.layer1.ff.weight.mul_(256 * 2**6)
                self.layer2.ff.weight.mul_(256 * 2**6)

        # axonal delay
        # TODO: implement delay layers for this
        self.buffer = deque(maxlen=1)
        self.buffer.append(torch.zeros(1, HID_SIZE))

    def forward(self, states, input_):
        out_states = [None] * len(states)

        out_states[0], s1 = self.layer1(states[0], input_)
        out_states[1], s2 = self.layer2(states[1], self.buffer[0])

        self.buffer.append(s1)

        return out_states, s2


def create_loihi(input_, weights):
    # network
    net = nx.NxNet()

    # spike generator
    sg = net.createSpikeGenProcess(numPorts=IN_SIZE)

    # prototypes
    compartment_proto = nx.CompartmentPrototype(
        vThMant=int(THRESH),
        compartmentCurrentDecay=int(LEAK_I),
        compartmentVoltageDecay=int(LEAK_V),
        biasMant=0,
        biasExp=0,
        functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
        # axondelay=0,
        # refractoryDelay=0,
    )
    connection_proto = nx.ConnectionPrototype(
        weight=1,
        weightExponent=0,
        disableDelay=True,
        delay=0,
        signMode=nx.SYNAPSE_SIGN_MODE.MIXED,
        postSynResponseMode=nx.SYNAPSE_POST_SYN_RESPONSE_MODE.EXPONENTIAL,
    )

    # create layers
    hidden_group = net.createCompartmentGroup(size=HID_SIZE, prototype=compartment_proto)
    output_group = net.createCompartmentGroup(size=OUT_SIZE, prototype=compartment_proto)

    # connect layers
    _ = sg.connect(
        hidden_group,
        prototype=connection_proto,
        weight=weights[0],
        # delay=np.zeros_like(wInHid.values),
    )
    _ = hidden_group.connect(
        output_group,
        prototype=connection_proto,
        weight=weights[1],
        # delay=np.zeros_like(wHidOut.values),
    )

    # add spikes
    for i in range(IN_SIZE):
        spike_times = torch.nonzero(input_[:, 0, i]).view(-1) + 1  # increment by 1; Loihi uses 1-based indexing
        sg.addSpikes(spikeInputPortNodeIds=i, spikeTimes=spike_times.tolist())

    i1, v1, s1 = hidden_group.probe(
        [nx.ProbeParameter.COMPARTMENT_CURRENT, nx.ProbeParameter.COMPARTMENT_VOLTAGE, nx.ProbeParameter.SPIKE]
    )
    i2, v2, s2 = output_group.probe(
        [nx.ProbeParameter.COMPARTMENT_CURRENT, nx.ProbeParameter.COMPARTMENT_VOLTAGE, nx.ProbeParameter.SPIKE]
    )

    return net, dict(i1=i1, v1=v1, s1=s1, i2=i2, v2=v2, s2=s2)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--loihi", action="store_true")
    args = parser.parse_args()

    # input spikes
    torch.manual_seed(SEED)
    input_spikes = torch.randint(2, (STEPS, 1, IN_SIZE)).float()

    # create torch
    with torch.no_grad():
        torch_net = SNN()

    # create loihi
    if args.loihi:
        import nxsdk.api.n2a as nx

        with torch.no_grad():
            weights = [torch_net.layer1.ff.weight.clone().div_(2**6).numpy(), torch_net.layer2.ff.weight.clone().div_(2**6).numpy()]
        loihi_net, loihi_data = create_loihi(input_spikes, weights)

    # run torch
    torch_data = dict(i1=[], v1=[], s1=[], i2=[], v2=[], s2=[])
    states = [None] * 2

    with torch.no_grad():
        for i in range(STEPS):

            # run net
            states, _ = torch_net(states, input_spikes[i])

            # log data
            i1, v1, s1 = states[0]
            i2, v2, s2 = states[1]
            torch_data["i1"].append(i1.numpy())
            torch_data["v1"].append(v1.numpy())
            torch_data["s1"].append(s1.numpy())
            torch_data["i2"].append(i2.numpy())
            torch_data["v2"].append(v2.numpy())
            torch_data["s2"].append(s2.numpy())

    # run loihi
    if args.loihi:
        loihi_net.run(STEPS + 1)  # increment by 1; Loihi uses 1-based indexing
        loihi_net.disconnect()

    # prepare data
    torch_data = {k: np.concatenate(v) for k, v in torch_data.items()}
    torch_data["s0"] = apply_nonzero(input_spikes.view(STEPS, IN_SIZE).numpy(), axis=1)
    torch_data["s1"] = apply_nonzero(torch_data["s1"], axis=1)
    torch_data["s2"] = apply_nonzero(torch_data["s2"], axis=1)

    if args.loihi:
        loihi_data["i1"] = loihi_data["i1"].data.T[1:, :]
        loihi_data["v1"] = loihi_data["v1"].data.T[1:, :]
        loihi_data["s1"] = apply_nonzero(loihi_data["s1"].data.T[1:, :], axis=1)
        loihi_data["i2"] = loihi_data["i2"].data.T[1:, :]
        loihi_data["v2"] = loihi_data["v2"].data.T[1:, :]
        loihi_data["s2"] = apply_nonzero(loihi_data["s2"].data.T[1:, :], axis=1)

    # plot
    # input and hidden
    fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    ax[0].eventplot(torch_data["s0"], linelengths=0.9)
    ax[0].set_ylabel("input #")
    ax[0].grid()
    ax[1].plot(torch_data["i1"])
    ax[1].set_ylabel("hidden current")
    ax[1].grid()
    ax[2].plot(torch_data["v1"])
    ax[2].set_ylabel("hidden voltage")
    ax[2].grid()
    ax[3].eventplot(
        torch_data["s1"], linelengths=0.9, color=[color_cycle[i % len(color_cycle)] for i in range(HID_SIZE)]
    )
    ax[3].set_ylabel("hidden #")
    ax[3].grid()

    fig.tight_layout()

    if args.loihi:
        # hidden
        ax[1].plot(loihi_data["i1"], "k:")
        ax[2].plot(loihi_data["v1"], "k:")
        ax[3].eventplot(loihi_data["s1"], linelengths=0.3, color="k")

    # output
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(torch_data["i2"])
    ax[0].set_ylabel("output current")
    ax[0].grid()
    ax[1].plot(torch_data["v2"])
    ax[1].set_ylabel("output voltage")
    ax[1].grid()
    ax[2].eventplot(
        torch_data["s2"], linelengths=0.9, color=[color_cycle[i % len(color_cycle)] for i in range(OUT_SIZE)]
    )
    ax[2].set_ylabel("output #")
    ax[2].grid()

    fig.tight_layout()

    if args.loihi:
        # hidden
        ax[0].plot(loihi_data["i2"], "k:")
        ax[1].plot(loihi_data["v2"], "k:")
        ax[2].eventplot(loihi_data["s2"], linelengths=0.3, color="k")

    plt.show()
