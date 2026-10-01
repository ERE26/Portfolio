# Publicar la web (para entrar desde cualquier sitio)

## Importante: Vercel NO sirve para esta web

Vercel está pensado para webs estáticas y funciones serverless (Next.js, etc.).
**Streamlit necesita un servidor Python que esté siempre encendido**, y eso Vercel
no lo da bien. Hay opciones mejores y gratis, pensadas justo para Streamlit.

## Opción recomendada: Streamlit Community Cloud (gratis)

Es de los creadores de Streamlit y despliega directo desde GitHub en 2 minutos.

1. Asegúrate de que el repo está en GitHub (ya lo está, rama
   `claude/magical-tesla-zoloss`).
2. Entra en **https://share.streamlit.io** y conecta tu cuenta de GitHub.
3. "New app" → elige el repo, la rama y el archivo **`app.py`**.
4. En **Advanced settings → Secrets**, pega tus claves (NO van en el repo):
   ```toml
   APCA_API_KEY_ID = "PK..."        # tu clave paper de Alpaca
   APCA_API_SECRET_KEY = "..."
   FRED_API_KEY = "..."             # opcional
   ```
   La app ya las lee de ahí (`_secrets_a_env`).
5. "Deploy". Te da una URL pública (p.ej. `tu-usuario-ilegalisimo.streamlit.app`)
   que abres desde el móvil o el ordenador.

**Qué funciona desplegado:**
- La vista **💼 Cartera demo (en vivo)** funciona entera: lee tu cuenta paper de
  Alpaca por API, muestra posiciones/P&L, oportunidades y noticias, y puedes
  maniobrar desde ahí. Solo necesita los Secrets de arriba.
- La vista **🔎 Buscador de patrones** necesita los datos históricos (`datos/`, que
  no van al repo por tamaño). Para que funcione en la nube hay que descargarlos en
  el arranque o subir un subconjunto. Si solo quieres la cartera en vivo, con la
  Cartera demo basta.

## Alternativas (también gratis, por si acaso)

- **Hugging Face Spaces** (plantilla Streamlit): https://huggingface.co/spaces →
  "Create Space" → SDK Streamlit → apunta a `app.py`, claves en *Settings → Secrets*.
- **Render** o **Railway**: despliegan el contenedor; comando de arranque
  `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

## Seguridad

- Las claves van SIEMPRE en el gestor de *Secrets* del hosting, **nunca** en el
  repositorio. `datos/.alpaca_key` está en `.gitignore`.
- Las claves son **paper** (empiezan por `PK`): la web nunca opera en real.
- La web puede colocar órdenes en tu cuenta paper (botones de maniobra). Si la
  publicas, cualquiera con el enlace podría tocarla: para una demo personal,
  Streamlit Cloud permite ponerla **privada** (solo tú), que es lo recomendable.
