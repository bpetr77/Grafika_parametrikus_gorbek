Feladat: Lagrange, Bézier és Catmull-Rom spline editort kell világ-koordinátarendszerben
megvalósítani. A teljes paramétertartomány minden görbe esetében a [0,1] intervallum. A Lagrange és
Catmull-Rom görbék ezt úgy osztják fel csomóértékekkel, hogy az egymás utáni csomóértékek különbsége
arányos legyen a kontrollpontok távolságával. A pontméret 10-es a vonalvastagság 2. A kontrollpontok
színe maximális intenzitású piros, a görbe pedig maximális intenzitású sárga. A fénykép (viewport) teljes
mértékben lefedi a 600x600 felbontású alkalmazói ablakot. A virtuális világban a távolság mértékegysége
[m], azaz méter.
Kezdetben a kameraablak közepe a világ-koordinátarendszer origójában van, mérete pedig 30x30 [m]. A
kameraablakot a felhasználó billentyűlenyomásokkal változtathatja, azt tologathatja (pan) és nagyíthat
(zoom):
’Z’: A kameraabla méretét 1.1-szeresére növeli a középpont megtartása mellett (zoom-out).
’z’: A kameraabla méretét 1/1.1-szeresére csökkenti a középpont megtartása mellett (zoom-in).
’P’: A kameraablakot jobbra 1 méterrel eltolja (pan).
’p’: A kameraablakot balra 1 méterrel eltolja (pan).
Ezen billentyűk megnyomásának hatására az aktuális görbe képe az új kameraablaknak megfelelően
azonnal változik.
A következőleg definiált görbe típusát az alábbi billentyűlenyomásokkal lehet meghatározni:
’l’: Lagrange
’b’: Bézier
’c’: Catmull-Rom
Ezen billentyűk lenyomásakor az aktuális görbe, ha létezik, megsemmisül és az új görbe
kontrollpontjainak megadásához kezdhetünk.
A görbe kontrollpontjai az egér bal gombjának lenyomásakor a kurzor alá kerülnek, azaz a bemeneti
csővezetéknek a kimeneti transzformáció inverzét kell produkálnia. Az egér jobb gombjának lenyomásával egy közeli (10 centiméternél közelebbi), már létező kontrollpont kiválasztható. A
kiválasztott kontrollpont addig követi a kurzort, amíg el nem engedjük a jobb gombot.
A ’T’ billentyűvel az aktuális és jövőbeli Catmull-Rom görbék tenzióparamétere 0.1-gyel növelhető, a ’t’
billentyűvel ugyanennyivel csökkenthető. A default tenzió zérus.
A kontrollpont és tenzióparaméter megváltoztatását a görbe alakja rögtön követi, azaz ilyenkor újra azt
kell rajzolni.
