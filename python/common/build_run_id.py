class Params:
    def __init__(self, rnn, n_obs, gamma, n_depth, vnet, net_size, n_episode, schedule, environment_type):
        self.rnn = rnn
        self.n_obs = n_obs
        self.gamma = gamma
        self.n_depth = n_depth
        self.vnet = vnet
        self.net_size = net_size
        self.n_episode = n_episode
        self.schedule = schedule
        self.environment_type = environment_type


def parse_value(line):
    return eval(line.split("=")[1].strip(" "))


def read_parameters(configfile):
    with open(configfile) as reader:
        lines = reader.readlines()
    for line in lines:
        if line.startswith("main.environment_type"):
            environment_type = parse_value(line)
        elif line.startswith("main.agent_name"):
            rnn = parse_value(line).startswith("rnn")
        elif line.startswith("main.episode_length"):
            n_episode = parse_value(line)

        elif line.startswith("environment_constructor.number_observations"):
            n_obs = parse_value(line)

        elif line.startswith("create_agent.gamma"):
            gamma = parse_value(line)
        elif line.startswith("create_agent.lstm_size"):
            n_depth = parse_value(line)[0]
        elif line.startswith("create_agent.value_fc_layer_params"):
            vnet_layers = parse_value(line)
        elif line.startswith("create_agent.fc_layer_params"):
            pnet_layers = parse_value(line)
        elif line.startswith("create_agent.use_learning_schedule"):
            schedule = parse_value(line)
        elif line.startswith("create_agent.decay_steps"):
            decay_steps = parse_value(line)
        elif line.startswith("create_agent.decay_rate"):
            decay_rate = parse_value(line)

    if pnet_layers == (100, 50):
        net_size = "normal"
    elif pnet_layers == (200, 100):
        net_size = "double"
    elif pnet_layers == (50, 25):
        net_size = "half"
    elif len(pnet_layers) > 2:
        net_size = "deep"
    else:
        net_size = "giant"

    if vnet_layers is None:
        vnet = "no"
    elif vnet_layers == (100,):
        vnet = None
    elif vnet_layers == (200,):
        vnet = "double"
    elif len(vnet_layers) > 2:
        vnet = "deep"
    else:
        vnet = "giant"

    if schedule and decay_rate == 0.9 and decay_steps == 1000:
        schedule = "normal"
    elif not schedule:
        schedule = "no"
    elif decay_rate > 0.9 or decay_steps > 1000:
        schedule = "slow"
    else:
        schedule = "fast"
    try:
        return Params(
            rnn=rnn,
            n_obs=n_obs,
            gamma=gamma,
            n_depth=n_depth,
            vnet=vnet,
            net_size=net_size,
            n_episode=n_episode,
            schedule=schedule,
            environment_type=environment_type
        )
    except UnboundLocalError as e:
        print("this config file is incomplete: " + configfile)
        print(e)


def build_run_id(params):
    run_id = "{}_obs".format(params.n_obs)
    if params.rnn:
        run_id += "_{}_depth".format(params.n_depth)
    if params.gamma != 0.99:
        run_id += "_{}_gamma".format(params.gamma)
    if params.vnet:
        run_id += f"_{params.vnet}_vnet"
    if not params.net_size == "normal":
        run_id += "_{}_pnet".format(params.net_size)
    if params.n_episode != 50:
        run_id += "_{}_episodes".format(params.n_episode)
    if not params.schedule == "normal":
        run_id += "_{}_schedule".format(params.schedule)
    run_id += "_{}_env".format(params.environment_type[:3])
    return run_id


def get_run_id(configfile):
    return build_run_id(read_parameters(configfile))
