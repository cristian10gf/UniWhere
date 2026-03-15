#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${SCRIPT_DIR}/data"

usage() {
	cat <<'EOF'
Uso: ./run-ace.sh <comando> <serie> [modelo.pt] [opciones] [-- args_extra]

Comandos:
  train <serie> [salida.pt]   Entrenar ACE. Si no se indica salida, la crea en
                               data/<serie>/ace/models/v<timestamp>.pt
  test  <serie> [modelo.pt]   Evaluar. Si no se indica modelo, usa el más reciente
                               en data/<serie>/ace/models/

La <serie> puede ser:
  block_g_1200          →  escena ACE: data/block_g_1200/ace
  block_g_1200/ace      →  idem (equivalente)

Si la escena ACE no existe, se convierte automáticamente desde COLMAP.

Opciones:
  --train-ratio R   Ratio train/test para colmap2ace (default: 0.8)
  --session NOMBRE  Sufijo para resultados del test (default: timestamp)
  --data-root PATH  Carpeta base de datos (default: preprocesamiento/data)
  --cpu             Forzar modo CPU
  -h, --help        Mostrar esta ayuda

Ejemplos:
  ./run-ace.sh train block_g_1200
  ./run-ace.sh train block_g_1200 -- --num_head_blocks 2
  ./run-ace.sh test  block_g_1200
  ./run-ace.sh test  block_g_1200 ace/models/v1_produccion.pt
  ./run-ace.sh test  block_g_1200 --session comparativa_v2
EOF
}

if [ $# -eq 0 ]; then
	usage
	exit 1
fi

COMMAND=""
SERIE=""
MODEL_ARG=""
TRAIN_RATIO="0.8"
SESSION_ARG=""
FORWARD_ARGS=()
SKIP_NEXT=0

for arg in "$@"; do
	if [ "$SKIP_NEXT" -eq 1 ]; then
		SKIP_NEXT=0
		continue
	fi

	case "$arg" in
		-h|--help)
			usage
			exit 0
			;;
		--train-ratio|--data-root|--session)
			SKIP_NEXT=1
			;;
	esac
done

# Parseo completo de argumentos
i=1
while [ $i -le $# ]; do
	arg="${!i}"
	case "$arg" in
		-h|--help)
			usage
			exit 0
			;;
		--train-ratio)
			i=$((i+1)); TRAIN_RATIO="${!i}"
			;;
		--data-root)
			i=$((i+1)); DATA_ROOT="${!i}"
			;;
		--session)
			i=$((i+1)); SESSION_ARG="${!i}"
			FORWARD_ARGS+=("--session" "${!i}")
			;;
		--)
			i=$((i+1))
			while [ $i -le $# ]; do
				FORWARD_ARGS+=("${!i}")
				i=$((i+1))
			done
			break
			;;
		-*)
			FORWARD_ARGS+=("$arg")
			;;
		*)
			if [ -z "$COMMAND" ]; then
				COMMAND="$arg"
			elif [ -z "$SERIE" ]; then
				SERIE="$arg"
			elif [ -z "$MODEL_ARG" ]; then
				MODEL_ARG="$arg"
			fi
			;;
	esac
	i=$((i+1))
done

if [ -z "$COMMAND" ] || [ -z "$SERIE" ]; then
	echo "Error: se requieren al menos <comando> y <serie>."
	echo ""
	usage
	exit 1
fi

case "$COMMAND" in
	train|test) ;;
	*)
		echo "Error: comando invalido '$COMMAND'. Usa 'train' o 'test'."
		exit 1
		;;
esac

DATA_ROOT="$(realpath "$DATA_ROOT")"

# --- Normalizar serie: block_g_1200 → block_g_1200/ace ---
# La serie siempre apunta a data/<serie>/ace como escena ACE.
SERIE_BASE="$SERIE"
if [[ "$SERIE" == */ace ]]; then
	SERIE_BASE="${SERIE%/ace}"
fi
ACE_SCENE_REL="${SERIE_BASE}/ace"
ACE_SCENE_FULL="${DATA_ROOT}/${ACE_SCENE_REL}"
MODELS_DIR="${ACE_SCENE_FULL}/models"

# --- Resolver ruta del modelo ---
resolve_model_path() {
	if [ -n "$MODEL_ARG" ]; then
		# Si el arg ya empieza con la serie o con data/, usarlo tal cual relativo a DATA_ROOT
		if [[ "$MODEL_ARG" == /* ]]; then
			# Ruta absoluta: hacerla relativa a DATA_ROOT
			realpath --relative-to="$DATA_ROOT" "$MODEL_ARG"
		elif [[ "$MODEL_ARG" == *.pt ]]; then
			# Ruta relativa: si no contiene la serie, asumir que es relativa a ace/models/
			if [[ "$MODEL_ARG" == */* ]]; then
				echo "${SERIE_BASE}/${MODEL_ARG}"
			else
				echo "${ACE_SCENE_REL}/models/${MODEL_ARG}"
			fi
		else
			echo "${SERIE_BASE}/${MODEL_ARG}"
		fi
		return
	fi

	if [ "$COMMAND" = "train" ]; then
		# Generar nombre con timestamp
		local ts
		ts="$(date +%Y%m%d_%H%M%S)"
		echo "${ACE_SCENE_REL}/models/v${ts}.pt"
	else
		# Test: buscar el .pt más reciente en models/
		if [ ! -d "$MODELS_DIR" ]; then
			echo "Error: no hay modelos entrenados en '$MODELS_DIR'." >&2
			echo "       Entrena primero con: ./run-ace.sh train ${SERIE_BASE}" >&2
			exit 1
		fi
		local latest
		latest=$(find "$MODELS_DIR" -maxdepth 1 -name "*.pt" -printf "%T@ %p\n" 2>/dev/null \
			| sort -n | tail -1 | cut -d' ' -f2-)
		if [ -z "$latest" ]; then
			echo "Error: no se encontro ningun modelo .pt en '$MODELS_DIR'." >&2
			echo "       Entrena primero con: ./run-ace.sh train ${SERIE_BASE}" >&2
			exit 1
		fi
		realpath --relative-to="$DATA_ROOT" "$latest"
	fi
}

MODEL_REL="$(resolve_model_path)"
MODEL_FULL="${DATA_ROOT}/${MODEL_REL}"

# --- Verificar/crear escena ACE ---
is_ace_scene_ready() {
	local scene_dir="$1"
	[ -d "$scene_dir/train/rgb" ]          || return 1
	[ -d "$scene_dir/train/poses" ]        || return 1
	[ -d "$scene_dir/train/calibration" ]  || return 1
	[ -d "$scene_dir/test/rgb" ]           || return 1
	[ -d "$scene_dir/test/poses" ]         || return 1
	[ -d "$scene_dir/test/calibration" ]   || return 1
	local n_train n_test
	n_train=$(find "$scene_dir/train/rgb" -maxdepth 1 -type f | wc -l)
	n_test=$(find  "$scene_dir/test/rgb"  -maxdepth 1 -type f | wc -l)
	[ "$n_train" -gt 0 ] || return 1
	[ "$n_test"  -gt 0 ] || return 1
	return 0
}

if ! is_ace_scene_ready "$ACE_SCENE_FULL"; then
	SOURCE_COLMAP_DIR="${DATA_ROOT}/${SERIE_BASE}"
	if [ ! -d "$SOURCE_COLMAP_DIR" ]; then
		echo "Error: serie COLMAP no encontrada en '$SOURCE_COLMAP_DIR'."
		exit 1
	fi
	echo "Escena ACE no encontrada. Convirtiendo desde COLMAP..."
	echo "  Origen : $SOURCE_COLMAP_DIR"
	echo "  Destino: $ACE_SCENE_FULL"
	echo "  Ratio  : $TRAIN_RATIO"
	uv run --with numpy --with scipy python "$SCRIPT_DIR/scripts/colmap2ace.py" \
		--colmap-dir "$SOURCE_COLMAP_DIR" \
		--output-dir "$ACE_SCENE_FULL" \
		--train-ratio "$TRAIN_RATIO"
	if ! is_ace_scene_ready "$ACE_SCENE_FULL"; then
		echo "Error: la conversion COLMAP->ACE no genero una escena valida."
		exit 1
	fi
fi

# Verificar modelo existente para test
if [ "$COMMAND" = "test" ] && [ ! -f "$MODEL_FULL" ]; then
	echo "Error: modelo no encontrado en '$MODEL_FULL'."
	exit 1
fi

exec "$SCRIPT_DIR/models/ace/docker/run.sh" \
	"$COMMAND" \
	"$ACE_SCENE_REL" \
	"$MODEL_REL" \
	--data-root "$DATA_ROOT" \
	"${FORWARD_ARGS[@]}"
