import requests
from colorama import Fore, Style


from epigraphdb_common_utils import neural_env_configs, docker_neural_env_configs
from app.es import es_client_connected


def check_env_configs() -> None:
    print(
        Style.BRIGHT
        + Fore.GREEN
        + "\n# Check environment configs"
        + Style.RESET_ALL
    )
    print(
        Style.BRIGHT
        + Fore.GREEN
        + "\n## Check environment configs for "
        + Fore.RED
        + "main_api server"
        + Style.RESET_ALL
    )
    print(
        Style.DIM + Fore.YELLOW + neural_env_configs.__doc__ + Style.RESET_ALL
    )
    print(neural_env_configs.env_configs)
    print(
        Style.BRIGHT
        + Fore.GREEN
        + "\n## Check environment configs for "
        + Fore.RED
        + "main_api docker container"
        + Style.RESET_ALL
    )
    print(
        Style.DIM + Fore.YELLOW + docker_neural_env_configs.__doc__ + Style.RESET_ALL
    )
    print(docker_neural_env_configs.env_configs)


def format_status(status: bool) -> str:
    if status:
        status_str = Style.BRIGHT + Fore.GREEN + str(status) + Style.RESET_ALL
    else:
        status_str = Style.BRIGHT + Fore.RED + str(status) + Style.RESET_ALL
    return status_str


def check_component_connections() -> None:
    print(
        Style.BRIGHT
        + Fore.GREEN
        + "\n# Check component connections"
        + Style.RESET_ALL
    )

    models_api_url = neural_env_configs.env_configs["models_api_url"]
    try:
        r = requests.get(f"{models_api_url}/ping")
        r.raise_for_status()
        models_api_connected = r.json()
        assert models_api_connected is True
    except:
        models_api_connected = False
    status = format_status(models_api_connected)
    print(f"models_api_url: {models_api_url}\tconnected: {status}")

    es_host = neural_env_configs.env_configs["es_host"]
    es_port = neural_env_configs.env_configs["es_port"]
    status = format_status(es_client_connected())
    print(f"elasticsearch: {es_host}:{es_port}\tconnected: {status}")


if __name__ == "__main__":
    check_env_configs()
    check_component_connections()
