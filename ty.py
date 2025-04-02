import streamlit as st
import qrcode
from PIL import Image
import io

# Titre de l'application
st.title("Générateur de QR Code Simple")

# Champ de saisie pour l'URL
site_url = st.text_input("URL du site (exemple: https://www.monsite.com)")

# Bouton pour générer le QR code
if st.button("Générer QR Code"):
    if site_url:
        # Ajout automatique du préfixe https:// si nécessaire
        if not (site_url.startswith('http://') or site_url.startswith('https://')):
            site_url = 'https://' + site_url
        
        # Création du QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=4)
        qr.add_data(site_url)
        qr.make(fit=True)
        
        # Création de l'image
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Conversion de l'image pour l'affichage
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        
        # Affichage du QR code
        st.image(buffered, caption=f"QR Code pour {site_url}")
        
        # Option de téléchargement
        st.download_button(
            label="Télécharger le QR Code",
            data=buffered.getvalue(),
            file_name="qrcode.png",
            mime="image/png"
        )
    else:
        st.warning("Veuillez entrer l'URL du site.")