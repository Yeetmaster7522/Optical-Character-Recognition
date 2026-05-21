#An impressively convoluted script to generate image files
#for every character, across a variety of fonts, with options
#to add rotation or noise.
#12SEN26

from PIL import Image, ImageDraw, ImageFont
import os
from random import randint

#Set which fonts and variations to use
complexity = 3      #1-4
noise = False
rotation = True
bold = True
italic = True

#Output settings
size = 28
img_size = (size, size)
font_sizes = {'Aclonica': 24, 'BowlbyOneSC': 24, 'FontdinerSwanky': 24, 'HomemadeApple': 18, 'Pacifico': 24}
#Default font size is 28pt; if a font needs a custom size, add it to the dictionary above
output_dir = "character_images"

def generateFontCharacters(fontname):
    print(f'Creating characters for {fontname}...')
    font_size = font_sizes.get(fontname, 28)
    for letter in characters:
        img = Image.new("L", img_size, color=255)
        draw = ImageDraw.Draw(img)

        myfont = fontname + "-"

        #NRBI[0] Random noise (±10%) added to each image
        #NRBI[1] Random angle (±20°) applied to each character
        #NRBI[2] Random characters (30% chance) are set to bold
        #NRBI[3] Random characters (30% chance) are set to italics

        NRBI = ['F','F','F','F']
        
        if bold and randint(1,100) <= 30:
            myfont += "Bold"
            NRBI[2] = 'T'
            
        if italic and randint(1,100) <= 30:
            myfont += "Italic"
            NRBI[3] = 'T'

        if NRBI[2] == NRBI[3] == 'F':
            myfont += "Regular"

        font_path = "12SEN CharacterGenerator/Fonts/" + folder + "/" + myfont + ".ttf"
        try:
            fontface = ImageFont.truetype(font_path, font_size)
        except:
            print(f'{font_path} does not exist; using regular')
            myfont = fontname + "-Regular"
            font_path = "Fonts/" + folder + "/" + myfont + ".ttf"
            fontface = ImageFont.truetype(font_path, font_size)
            NRBI[2] = 'F'
            NRBI[3] = 'F'


        # ChatGPT did maths - gets the size of the text to position it at the centre of the image
        bbox = draw.textbbox((0, 0), chr(letter), font=fontface)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        position = ((0-bbox[0]+((size-text_width)//2)),(0-bbox[1]+((size-text_height)//2)))

        draw.text(position, chr(letter), fill=False, font=fontface)
        

        if rotation:
            r = randint(-20,20)//5*5
            if r != 0:
                img = img.rotate(r,resample=Image.Resampling.BICUBIC,fillcolor="white")
                NRBI[1] = 'T'

        if noise:
            for i in range(75):
                x = randint(0,27)
                y = randint(0,27)
                current = img.getpixel((x,y))
                invert = 255 - current
                img.putpixel((x,y),invert)
            NRBI[0] = 'T'
        
        img.save(os.path.join(output_dir, f"{myfont}_{''.join(NRBI)}_{letter}.png"))

#Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

#Fonts
#To generate characters for additional fonts:
# - The filename must be "{fontface}-Regular.ttf" (exactly)
# - The font file must be in the appropriate subfolder of Fonts
# - Add the fontface to the appropriate list below
serifList = ['BreeSerif', 'EBGaramond', 'Georgia', 'PalatinoLinotype', 'Merriweather', 'TimesNewRoman']
sansList = ['Arial', 'Calibri', 'Comfortaa', 'Montserrat', 'Oxygen', 'Verdana']
monoList = ['Consolas', 'CourierNew', 'GoogleSansCode', 'RobotoMono', 'SourceCodePro']
deco1List = ['Aclonica', 'Algerian', 'BowlbyOneSC', 'ComicSansMS', 'PermanentMarker', 'SairaStencil']
deco2List = ['Caveat', 'Creepster', 'FontdinerSwanky', 'HomemadeApple', 'Pacifico', 'Yellowtail']

#charlist
characters= [i for i in range(65,91)]     #uppercase
characters += [i for i in range(97,123)]   #lowercase
characters += [i for i in range(48,58)]    #numbers
if complexity >= 3:
    characters += [ord('@'), ord('#'), ord('$'), ord('%'), ord('&'), ord('+'), ord('?'), ord('<'), ord('>')]


#Iterate through the fonts in their subfolders
folderList = ['Monospace','Sans','Serif']
if complexity >= 3:
    folderList.append('Decorative')

fontList = {'Monospace': monoList, 'Sans': sansList, 'Serif': serifList, 'Decorative': deco1List}
if complexity >= 4:
    fontList['Decorative'] += deco2List

for folder in folderList:
    for fontname in fontList[folder]:
        try:
            generateFontCharacters(fontname)
        except Exception as e:
            print(e)