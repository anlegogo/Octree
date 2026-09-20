# Objetivo específico 2 - HCE, SVM y Bosque Aleatorio

Este flujo define y valida las características manuales extraídas del octree
antes de entrenar los clasificadores tradicionales. La selección de columnas
se realiza exclusivamente con el split oficial `train`; `test` no participa en
la auditoría, el escalado ni el ajuste de hiperparámetros.

## Prerrequisito

Los 12 311 modelos de ModelNet40 deben haber sido regenerados con el formato
definitivo del octree y la auditoría estricta del objetivo 1 debe terminar sin
errores:

```bash
python Scripts_Analisis/medir_metricas_off.py \
  --manifests-dir data/manifests \
  --output-root data \
  --resultados-dir resultados/objetivo1
```

No se debe utilizar `--permitir-incompleto` en esta verificación.

## 1. Auditoría de características sobre train

```bash
python fase3_hce/auditar_train_hce.py \
  --data-root data \
  --resultados-dir resultados/objetivo2 \
  --resoluciones 32 64
```

Para cada resolución, el script valida:

- los 9 843 NPZ del split `train`;
- formato, versión, resolución, profundidad, clase, etiqueta y `model_id`;
- dimensión y orden del vector HCE base;
- ausencia de `NaN` e `Inf`;
- características constantes calculadas únicamente con `train`.

Se generan cuatro archivos versionables:

```text
resultados/objetivo2/auditoria_hce_train_R32.json
resultados/objetivo2/auditoria_hce_train_R64.json
resultados/objetivo2/contrato_hce_R32.json
resultados/objetivo2/contrato_hce_R64.json
```

Un contrato solo habilita el entrenamiento cuando contiene
`"autorizado_para_entrenamiento": true`. Las columnas constantes quedan
excluidas y su motivo se registra explícitamente. Una auditoría parcial puede
ejecutarse con `--limite N --permitir-incompleto`, pero nunca autoriza SVM ni
Bosque Aleatorio.

## 2. Pruebas automatizadas

```bash
python -m pip install -r requirements-objetivo2.txt
python -m pytest
```

## 3. Entrenamiento

Solo después de aprobar ambos contratos:

```bash
python fase3_hce/fase3_hce_entrenamiento.py \
  --resolucion 32 \
  --data-root data \
  --contrato-features resultados/objetivo2/contrato_hce_R32.json \
  --resultados-dir resultados/objetivo2

python fase3_hce/fase3_hce_entrenamiento.py \
  --resolucion 64 \
  --data-root data \
  --contrato-features resultados/objetivo2/contrato_hce_R64.json \
  --resultados-dir resultados/objetivo2
```

La reserva interna de validación es estratificada, reproducible y usa la
semilla 42. El SVM aplica `StandardScaler` dentro del `Pipeline` de validación
cruzada. El conjunto oficial `test` se carga únicamente después de aprobar el
contrato de características generado con `train`.
