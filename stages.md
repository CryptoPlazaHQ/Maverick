¡Excelente idea! Para que tengas la lógica Maverick lista para **usar como checklist** o **tabla de validación rápida**, te lo estructuré así:

---

# ✅ Checklist de Validación Maverick

| **Fase**               | **Qué ocurre**          | **Condición para avanzar**                                              | **Invalidación**                                               | **Notas**                                                                             |
| ---------------------- | ----------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **STANDBY**            | No hay ciclo activo     | Detectar pivote inicial: <br>• LONG: suelo (sL) <br>• SHORT: techo (sH) | –                                                              | Ambos sesgos (LONG/SHORT) activos en paralelo                                         |
| **SEED\_P1**           | Se fija P1              | Aparece pivote opuesto: <br>• LONG: techo (P0) <br>• SHORT: suelo (P0)  | P1 reemplazado si aparece uno más extremo                      | P0 todavía dinámico                                                                   |
| **PROVISIONAL\_P0**    | P0 definido provisional | Retroceso toca banda \[0.382 – 0.786] (HL+tol)                          | • Si rompe P1 <br>• Si retroceso pasa 0.786 antes del breakout | P0 puede actualizarse hasta validar banda                                             |
| **VALIDATE\_P0**       | P0 validado             | Precio rompe contra P0 en dirección del sesgo (HL+tol)                  | • Romper P1 <br>• Cruzar 0.786 antes del breakout              | Ahora P0 es fijo                                                                      |
| **BREAKOUT\_1**        | Se confirma breakout    | Precio rompe P0 en dirección sesgo                                      | • Romper P1 <br>• Pasarse de 0.786 (si aún no hay lock)        | Aquí puede activarse **EarlyInvalidationLock** si retroceso revierte breakout1        |
| **PULLBACK\_2**        | Retroceso post-breakout | Retrocede dentro de la banda \[0.382–0.786]                             | • Si NO hay Early Lock: tocar/pasar 0.786 invalida             | Con Early Lock → ignora invalidación 0.786                                            |
| **BREAKOUT\_2** (Lock) | Segundo breakout        | Precio supera de nuevo nivel breakout1 en dirección sesgo               | –                                                              | Patrón bloqueado (**patternLocked = TRUE**) <br> Se calculan zonas (Operative, Entry) |
| **UBCT\_CYCLING**      | Modo operativo activo   | Ciclo sigue                                                             | –                                                              | Se esperan zonas activas hasta el fin                                                 |
| **CYCLE\_END**         | Fin del ciclo           | • Tocar Copy 0.236 <br>• Timeout de barras                              | –                                                              | Se guarda resumen y vuelve a STANDBY                                                  |

---

## 🔑 Resumen rápido de invalidaciones

* **Romper P1** → ciclo cancelado, volver a STANDBY (nuevo P1 después del anterior).
* **0.786** → invalida en PROVISIONAL\_P0, VALIDATE\_P0, BREAKOUT\_1 y PULLBACK\_2 (salvo Early Lock).
* **Early Lock** → solo se evalúa tras BREAKOUT\_1 si el retroceso perfora breakout1 en reversa.
* **Lock** en BREAKOUT\_2 → el ciclo ya no se “desbloquea”, solo finaliza en CYCLE\_END.

---

## 🎯 Usabilidad

* Cada fila es un paso: puedes ir marcando ✅ cuando se cumpla la condición.
* Si ocurre una invalidación → vuelves a STANDBY y sigues con el siguiente pivote.
* Esta tabla funciona como **checklist en vivo** cuando hagas trazados manuales o debugging en la app Streamlit.

---

¿Quieres que te arme también una **versión imprimible en PDF** con este checklist (con espacio para que marques a mano P1, P0, Pullbacks, etc.), o prefieres mantenerlo digital en la app?
