# Resultados del objetivo específico 1

Esta carpeta recibirá exclusivamente los resultados producidos por el flujo
documentado en el `README.md` principal.

Los archivos históricos ubicados directamente en `resultados/` fueron
generados con versiones anteriores del pipeline y no demuestran por sí solos
el cumplimiento del objetivo específico 1.

La evidencia para aceptar la implementación del objetivo específico 1 debe
incluir, como mínimo:

- `validacion_chair_0001.json`;
- `muestra_controlada/resumen_ejecucion.json`, sin
  modelos fallidos;
- `muestra_controlada/metricas_muestra_controlada.csv`;
- `muestra_controlada/metricas_muestra_controlada_verificadas.csv`;
- `muestra_controlada/resumen_metricas_muestra_controlada.json` sin errores de
  integridad o equivalencia.

La regeneración de los 12 311 modelos de ModelNet40 se conserva como una
etapa posterior, necesaria antes de extraer las características y entrenar los
clasificadores, pero no como requisito para validar la implementación del
octree.
