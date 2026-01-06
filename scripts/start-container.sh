#!/bin/bash
set -e

usage() {
    echo "Usage: $0 [OPTION]..."
    echo ""
    echo "Optional arguments:"
    echo "  -h, --help     Give this help list"
    echo "      --image    Select the image name of the container"
    echo "  -v, --volume   Bind mount a volume into the container"
    echo "      --name     Assign a name to the container"
    echo "  -d, --detach   Run container in background and print container ID"
    echo "  -H, --history  Record the container bash history"
    exit 1
}

image_name="maq:latest"  # 請將此處替換為你構建的映像名稱
container_tool=$(basename $(which podman) 2>/dev/null)

if [[ ! $container_tool ]]; then
    echo "Neither podman nor docker is installed." >&2
    exit 1
fi

$container_tool build -t maq .

container_volume="-v .:/workspace"
container_arguments="--gpus all"
record_history=true

while :; do
    case $1 in
        -h|--help) shift; usage ;;
        --image) shift; image_name=${1} ;;
        -v|--volume) shift; container_volume="${container_volume} -v ${1}" ;;
        --name) shift; container_arguments="${container_arguments} --name ${1}" ;;
        -d|--detach) container_arguments="${container_arguments} -d" ;;
        -H|--history) record_history=true ;;
        "") break ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
    shift
done

if [ "$record_history" = true ]; then
    history_dir=".container_root"
    if [ ! -d ${history_dir} ]; then
        mkdir -p ${history_dir}
        $container_tool run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --rm -it -v "${history_dir}:/container_root" ${image_name} /bin/bash -c "cp -r /root/. /container_root && touch /container_root/.bash_history && exit"
    fi
    container_volume="${container_volume} -v ${history_dir}:/root"
fi

container_arguments=$(echo ${container_arguments} | xargs)
echo "$container_tool run ${container_arguments} --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --network=host --ipc=host --rm -it ${container_volume} ${image_name}"
$container_tool run ${container_arguments} --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --network=host --ipc=host --rm -it ${container_volume} ${image_name}
