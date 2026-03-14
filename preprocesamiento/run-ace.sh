#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
	cat <<'EOF'
Uso: ./run-ace.sh <comando> <escena> <salida_o_modelo> [opciones] [-- args_extra]

Wrapper de ACE con integración COLMAP->ACE automática:
- Si la escena ACE indicada no está creada/completa, ejecuta colmap2ace.py antes de correr ACE.
- La conversión se ejecuta con uv.

Opciones adicionales de este wrapper:
  --train-ratio R   Ratio train/test para colmap2ace (default: 0.8)

Opciones reenviadas al runner ACE (run.sh):
  --data-root PATH
  --cpu
  ...
EOF
}

if [ $# -eq 0 ]; then
	usage
	exit 1
fi

COMMAND=""
SCENE_PATH=""
MODEL_PATH=""
DATA_ROOT=""
TRAIN_RATIO="0.8"

# Parse minimally to locate scene/data-root for auto-conversion.
for ((i = 1; i <= $#; i++)); do
	arg="${!i}"

	case "$arg" in
		-h|--help)
			usage
			exit 0
			;;
		--data-root)
			if [ $((i + 1)) -le $# ]; then
				next_index=$((i + 1))
				DATA_ROOT="${!next_index}"
			fi
			;;
		--train-ratio)
			if [ $((i + 1)) -le $# ]; then
				next_index=$((i + 1))
				TRAIN_RATIO="${!next_index}"
			fi
			;;
	esac

	if [[ "$arg" != -* ]]; then
		if [ -z "$COMMAND" ]; then
			COMMAND="$arg"
		elif [ -z "$SCENE_PATH" ]; then
			SCENE_PATH="$arg"
		elif [ -z "$MODEL_PATH" ]; then
			MODEL_PATH="$arg"
		fi
	fi
done

if [ -z "$DATA_ROOT" ]; then
	DATA_ROOT="${SCRIPT_DIR}/data"
fi
DATA_ROOT="$(realpath "$DATA_ROOT")"

if [ -z "$COMMAND" ] || [ -z "$SCENE_PATH" ] || [ -z "$MODEL_PATH" ]; then
	# Keep original behavior and error handling in downstream runner.
	exec "$SCRIPT_DIR/models/ace/docker/run.sh" "$@"
fi

FORWARD_ARGS=()
SKIP_NEXT=0
for arg in "$@"; do
	if [ "$SKIP_NEXT" -eq 1 ]; then
		SKIP_NEXT=0
		continue
	fi

	if [ "$arg" = "--train-ratio" ]; then
		SKIP_NEXT=1
		continue
	fi

	FORWARD_ARGS+=("$arg")
done

SCENE_FULL="${DATA_ROOT}/${SCENE_PATH}"

is_ace_scene_ready() {
	local scene_dir="$1"

	[ -d "$scene_dir/train/rgb" ] || return 1
	[ -d "$scene_dir/train/poses" ] || return 1
	[ -d "$scene_dir/train/calibration" ] || return 1
	[ -d "$scene_dir/test/rgb" ] || return 1
	[ -d "$scene_dir/test/poses" ] || return 1
	[ -d "$scene_dir/test/calibration" ] || return 1

	local train_rgb_count
	local test_rgb_count
	train_rgb_count=$(find "$scene_dir/train/rgb" -maxdepth 1 -type f | wc -l)
	test_rgb_count=$(find "$scene_dir/test/rgb" -maxdepth 1 -type f | wc -l)
	[ "$train_rgb_count" -gt 0 ] || return 1
	[ "$test_rgb_count" -gt 0 ] || return 1

	return 0
}

infer_colmap_source_scene() {
	local ace_scene_rel="$1"

	if [[ "$ace_scene_rel" == */ace ]]; then
		echo "${ace_scene_rel%/ace}"
		return
	fi

	if [[ "$ace_scene_rel" == ace ]]; then
		echo ""
		return
	fi

	# Fallback: assume user passed a base scene and wants output under <scene>/ace.
	echo "$ace_scene_rel"
}

if ! is_ace_scene_ready "$SCENE_FULL"; then
	SOURCE_SCENE_REL="$(infer_colmap_source_scene "$SCENE_PATH")"

	if [ -z "$SOURCE_SCENE_REL" ]; then
		echo "Error: no se pudo inferir escena COLMAP origen desde '$SCENE_PATH'."
		echo "Usa una ruta de escena ACE terminada en '/ace' (por ejemplo: serie-1/ace)."
		exit 1
	fi

	SOURCE_COLMAP_DIR="${DATA_ROOT}/${SOURCE_SCENE_REL}"
	if [ ! -d "$SOURCE_COLMAP_DIR" ]; then
		echo "Error: escena COLMAP origen no encontrada en '$SOURCE_COLMAP_DIR'."
		exit 1
	fi

	echo "Escena ACE no encontrada/incompleta. Ejecutando conversión COLMAP->ACE..."
	echo "  Origen : $SOURCE_COLMAP_DIR"
	echo "  Destino: $SCENE_FULL"
	echo "  Ratio  : $TRAIN_RATIO"

	uv run --with numpy --with scipy python "$SCRIPT_DIR/scripts/colmap2ace.py" \
		--colmap-dir "$SOURCE_COLMAP_DIR" \
		--output-dir "$SCENE_FULL" \
		--train-ratio "$TRAIN_RATIO"

	if ! is_ace_scene_ready "$SCENE_FULL"; then
		echo "Error: la conversión COLMAP->ACE no generó una escena ACE válida en '$SCENE_FULL'."
		exit 1
	fi
fi

exec "$SCRIPT_DIR/models/ace/docker/run.sh" "${FORWARD_ARGS[@]}"
