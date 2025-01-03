from deep_translator import GoogleTranslator

# Texto a ser traduzido
texto_original = "Qual o diagrama elétrico recomendado para conectar um motor trifásico WEG de 15 HP?"

# Traduzindo para o inglês
traducao = GoogleTranslator(source='auto', target='en').translate(texto_original)

print(traducao)