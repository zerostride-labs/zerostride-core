# ZeroStride

ZeroStride es un motor de inferencia experimental en C++ enfocado en la ejecución de modelos Mixture-of-Experts (MoE) en hardware con restricciones de memoria física.

El objetivo del proyecto es investigar y evaluar técnicas para mitigar la variabilidad en los tiempos de generación provocada por la paginación del sistema operativo (*page faults*), explorando:

- Asignación estática de memoria y eliminación de reservas dinámicas en el bucle de inferencia.
- Streaming predictivo y asíncrono de pesos de expertos desde almacenamiento secundario.
- Uso de memoria acotada y determinista durante la decodificación.

Proyecto inspirado en herramientas del ecosistema abierto como `llama.cpp` y `strata`.

## Estado

Fase temprana de diseño e investigación técnica.

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.
