# Práctica 1 — Clasificación con un MLP: respuestas

Respuestas a las nueve preguntas del enunciado. Todo el código, las gráficas y la justificación de cada paso están en `practica_1_resuelta.ipynb`.

Los números son los de este notebook. Si al ejecutarlo en otro entorno cambia alguna cifra decimal, las conclusiones son las mismas.

**1. ¿Qué transformaciones has hecho sobre los datos? ¿Has eliminado alguna columna? ¿Cuál y por qué?**

- He eliminado **la columna `Date`**. Es el identificador de la fila, no una medida del tiempo (como `Name` en el Titanic). Con una partición aleatoria, la red podría aprender "qué tiempo hizo esa semana" en lugar de relaciones que generalicen, y la estación del año ya la recogen las temperaturas. Solo la he usado aparte para comprobar la partición.
- He eliminado **40 filas** (0.6 %), las de los días en que no se midió la lluvia (`Rainfall` y `RainToday` vacíos a la vez). No hay duplicados que quitar.
- `RainToday` y `RainTomorrow`: de `Yes`/`No` a 1/0 (`map`, Tema 1).
- Partición en train/validación/test (56 % / 14 % / 30 %: primero el 30 % de test y luego el 20 % del resto para validación), estratificada por estación y objetivo, **antes** de cualquier transformación.
- Imputación de las numéricas con la **mediana de train de cada estación**: los ausentes se concentran en Canberra, en `Evaporation`, `Sunshine` y `Cloud`.
- `WindGustDir`: los ausentes como categoría `"desconocida"` y `OneHotEncoder(drop="first")`, porque no tiene orden. `Location`: una sola columna binaria con el mismo *encoder*.
- `StandardScaler` sobre las 14 numéricas, ajustado con train y aplicado a los tres conjuntos. Las binarias se quedan en 0/1.
- La imputación, el *encoder* y el escalador se **ajustan solo con train**. Resultado: 32 entradas.

**2. ¿Qué funciones de activación has elegido para las capas ocultas y para la capa de salida?**

**ReLU** en las capas ocultas: es la activación oculta de todas las redes del curso, es barata, y su derivada vale 1 en la parte activa, así que no atenúa el gradiente al propagarlo hacia atrás. **Sigmoide** en la neurona de salida: es una clasificación binaria, y la sigmoide da un número en $(0, 1)$ que se interpreta como $P(\text{llueve mañana})$ y se umbraliza en 0.5 (tabla tarea → activación de salida → coste, p. 15). Una salida lineal no da una probabilidad, y una ReLU en la salida puede morir (Tema 2, clase 2).

**3. ¿Qué función de coste has utilizado? ¿Qué métrica de evaluación has escogido?**

Coste: **entropía cruzada binaria** (`binary_crossentropy`), la que acompaña a la salida sigmoide (Tema 4, clase 1). Con la sigmoide, su gradiente respecto a $z$ es simplemente $a - y$ (pregunta 8). Con el MSE aparecería un factor $\sigma'(z)$ que tiende a 0 cuando la sigmoide se satura, justo cuando la red se equivoca con mucha confianza.

Métrica: **accuracy**, como en el Tema 1, siempre comparada con los baselines (0.755 la mayoritaria y 0.767 la persistencia): con un 25 % de días de lluvia, una *accuracy* suelta no dice nada. La complemento con la **matriz de confusión**, que muestra que la red acierta el 93 % de los días secos y el 58 % de los de lluvia.

**4. Compara el resultado de un perceptrón simple frente a una red neuronal.**

En validación, con cinco semillas: el perceptrón tiene mediana **0.777** y rango de 0.670 a 0.808, 14 puntos según el orden de las muestras, y con una semilla queda por debajo de la clase mayoritaria. El MLP elegido tiene mediana **0.845** y rango de 0.839 a 0.846. Las barras no se tocan, así que la diferencia es real. En test: perceptrón 0.809 y MLP 0.840.

El perceptrón solo traza un hiperplano y su regla solo corrige cuando falla, así que con clases que se solapan no converge. El MLP añade una capa oculta no lineal y minimiza la entropía cruzada con descenso de gradiente.

**5. ¿Cuántas capas y cuántas neuronas por capa tiene? ¿Cómo has llegado a ella?**

**2 capas ocultas de 32 neuronas ReLU** y una salida sigmoide: 32 → 32 → 32 → 1, 2145 parámetros. He barrido 1 capa con 2, 4, 8, 16, 32 y 64 neuronas y 2 capas con 8, 16 y 32, todas con el mismo presupuesto (50 *epochs*, $\eta = 0.01$, $B = 32$) y **tres semillas por punto**. He elegido por la mediana de la *accuracy* de validación y, entre las topologías empatadas dentro del rango de semillas de la mejor (1 × 64 y 2 × 32), la de menos parámetros.

La curva muestra *underfitting* con 2–4 neuronas y un aplanamiento a partir de 16: de 1 × 16 a 2 × 32 hay medio punto. Lo que se sostiene es que una red pequeña de una o dos capas basta y que más capacidad no aporta.

**6. ¿Cuál es el error o la accuracy que obtienes sobre los datos de entrenamiento, de validación y de test? ¿Óptima, *overfitting* o *underfitting*?**

*Accuracy*: **train 0.862, validación 0.845, test 0.840**.

- **No hay *overfitting* apreciable.** La brecha train–validación es de 1.7 puntos y el test confirma la validación. Lo he evitado parando en la *epoch* 93: entrenando hasta la 300, la pérdida de validación sube de 0.345 a 0.360 mientras la de train sigue bajando, y eso sí sería sobreajuste.
- **Tampoco *underfitting* por falta de capacidad.** Las redes más grandes del barrido no mejoran en validación, y la *accuracy* de train apenas sube con la capacidad (de 0.85 a 0.86).
- **El techo del 84–86 % viene de los datos**, no de la red. Con las medidas de hoy, parte de la lluvia de mañana es impredecible: el 42 % de los días de lluvia de validación quedan por debajo de 0.5 de probabilidad. Es el ruido irreducible.

Dentro de lo que se puede hacer con estas variables y SGD, la configuración está cerca de lo óptimo: las diferencias con las alternativas razonables caben en el ruido entre semillas.

**7. ¿Cuántas *epochs* y cómo las has elegido? Gráfica de la función de coste. ¿Qué otra gráfica podrías utilizar?**

**93 *epochs***. He entrenado la topología elegida 300 *epochs* con la validación en cada una (gráfica en la sección *El número de epochs*). Las dos curvas bajan juntas hasta la *epoch* 93, donde la pérdida de validación toca su mínimo (0.345). Desde ahí sube mientras la de train sigue bajando, así que he tomado ese mínimo (*early stopping* a mano, como `mejor_epoch` en el Tema 3, clase 2) y he reentrenado hasta él con cinco semillas.

Otra gráfica: la **matriz de confusión** de validación (`ConfusionMatrixDisplay`, Tema 1), que separa los aciertos por clase: 93 % de los días secos y 58 % de los de lluvia. También el **histograma de la probabilidad predicha** separado por la clase real, que muestra si la red separa las dos clases y dónde se solapan.

**8. ¿Qué derivadas de funciones de activación necesitas? Derívalas.**

Dos, desarrolladas en la sección de las derivadas y comprobadas numéricamente allí mismo:

- **ReLU**: $h(z) = \max(0, z)$, $\;h'(z) = 1$ si $z > 0$ y $0$ si $z < 0$ (en $z = 0$ se toma 0).
- **Sigmoide**: $\sigma(z) = (1 + e^{-z})^{-1}$, $\;\sigma'(z) = \dfrac{e^{-z}}{(1+e^{-z})^2} = \sigma(z)\,(1 - \sigma(z))$.

Con la entropía cruzada binaria, $\dfrac{\partial \mathcal{L}}{\partial z_{\text{salida}}} = \dfrac{a-y}{a(1-a)}\cdot a(1-a) = a - y$. Ese error se propaga hacia atrás multiplicando por los pesos y por $h'(z)$ en cada capa ReLU.

**9. ¿Cómo has tratado la columna `Location`? ¿Cómo rinde tu red en cada estación, y con qué comparas cada resultado?**

`Location` es categórica sin orden con dos valores: `OneHotEncoder(drop="first")` la deja en una columna binaria, `Location_Canberra`. Además, la imputación de ausentes se hace con las medianas de train **de cada estación**, y la partición está estratificada por estación y objetivo.

Cada estación se compara con **sus propios baselines**, calculados con sus días: su clase mayoritaria y su persistencia. En test:

| | días | llueve al día siguiente | MLP | mayoritaria | persistencia | MLP − mejor baseline |
|---|---|---|---|---|---|---|
| Cairns | 890 | 31.6 % | **0.811** | 0.684 | 0.755 | +5.6 |
| Canberra | 1020 | 18.3 % | **0.865** | 0.817 | 0.769 | +4.8 |

Sin baselines parecería que la red funciona mucho mejor en Canberra, pero allí el listón es más alto. En Cairns el mejor baseline es la persistencia (la lluvia viene en rachas) y en Canberra la clase mayoritaria (es más seca), y la red mejora a los dos en una magnitud parecida. Quitando la columna, la red rinde igual (mediana 0.845 en validación): reconstruye la estación a partir de las temperaturas (mínima media de 21 °C en Cairns frente a 7 °C en Canberra). Es una variable proxy, como en el Tema 2.
