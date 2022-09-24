class Params:
    def __init__(
            self,
            environment_type,
            agent_name,
            rnn,
            input_dimension,
            function_name,
            number_free_parameters,
            episode_length,

            batch_size,

            number_observations,
            randomize_start,

            n_start_pos,

            gamma,
            n_depth,
            vnet,
            net_size,
            schedule):
        self.environment_type = environment_type
        self.agent_name = agent_name
        self.rnn = rnn
        self.input_dimension = input_dimension
        self.function_name = function_name
        self.number_free_parameters = number_free_parameters
        self.episode_length = episode_length

        self.batch_size = batch_size

        self.number_observations = number_observations
        self.randomize_start = randomize_start

        self.n_start_pos = n_start_pos

        self.gamma = gamma
        self.n_depth = n_depth
        self.vnet = vnet
        self.net_size = net_size
        self.schedule = schedule


def parse_value(line):
    return eval(line.split("=")[1].strip(" "))


def read_parameters(configfile):
    with open(configfile) as reader:
        lines = reader.readlines()
    for line in lines:
        if line.startswith("main.environment_type"):
            environment_type = parse_value(line)
        elif line.startswith("main.agent_name"):
            agent_name = parse_value(line)
            rnn = agent_name.startswith("rnn")
        elif line.startswith("main.input_dimension"):
            input_dimension = parse_value(line)
        elif line.startswith("main.function_name"):
            function_name = parse_value(line)
        elif line.startswith("main.number_free_parameters"):
            number_free_parameters = parse_value(line)
        elif line.startswith("main.episode_length"):
            episode_length = parse_value(line)

        elif line.startswith("train.batch_size"):
            batch_size = parse_value(line)

        elif line.startswith("environment_constructor.number_observations"):
            number_observations = parse_value(line)
        elif line.startswith("environment_constructor.randomize_start"):
            randomize_start = parse_value(line)

        elif line.startswith("evaluation_driver_init.n_start_pos"):
            n_start_pos = parse_value(line)

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
            environment_type=environment_type,
            agent_name=agent_name,
            rnn=rnn,
            input_dimension=input_dimension,
            function_name=function_name,
            number_free_parameters=number_free_parameters,
            episode_length=episode_length,

            batch_size=batch_size,

            number_observations=number_observations,
            randomize_start=randomize_start,

            n_start_pos=n_start_pos,

            gamma=gamma,
            n_depth=n_depth,
            vnet=vnet,
            net_size=net_size,
            schedule=schedule
        )
    except UnboundLocalError as e:
        print("this config file is incomplete: " + configfile)
        print(e)


def build_run_id(params):
    run_id = "{}_{}_env_{}_epsLen_{}_numObs".format(
        params.environment_type[:3],
        params.agent_name,
        params.episode_length,
        params.number_observations
    )
    if params.rnn:
        run_id += "_{}_depth".format(params.n_depth)
    if params.gamma != 0.99:
        run_id += "_{}_gamma".format(params.gamma)
    if params.vnet:
        run_id += f"_{params.vnet}_vnet"
    if not params.net_size == "normal":
        run_id += "_{}_pnet".format(params.net_size)
    if not params.schedule == "normal":
        run_id += "_{}_schedule".format(params.schedule)
    return run_id


def get_run_id(configfile):
    return build_run_id(read_parameters(configfile))
