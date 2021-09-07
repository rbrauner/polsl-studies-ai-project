# Projekt AI

## Autorzy

- Rafał Brauner
- Dawid Piela

## Sprzęt i środowisko

Na początku korzystaliśmy z pythona, następnie przenieśliśmy się na Jupitera, ze względu na wygodę i możliwość odpalania kolejnych etapów programu po koleii, w przeciwieństwie do czytego pythona, gdzie trzeba było uruchomić cały program. Po paru problemach związanych z użyciem karty graficznej, przenieśliśmy się na platformę Google Colabs, który posiadał 2 zalety:

- pierwszą była łatwiejsze przestawienie na użycie karty graficznej
- drugą była lepsza wydajność karty graficznej

Do przechowywania danych wykorzystaliśmy dysk Google.

## Info

Nasz projekt rozpoczeliśmy od zapoznaniem się z sztuczną inteligencją i uczeniem maszynowym.

Do pierwszych kroków wzieliśmy zestaw danych z obrazkami psów, kotów oraz pand. Pobraliśmy dane treningowe i testowe poprzez funckję `train_test_split` i stworzyliśmy pierwszy model `U-Net`. Pod koniec wykorzystaliśmy wytrenowany model na danych testowych i wyświetliliśmy wykres przedstawiający ile obrazków z zbioru testowego zostało rozpoznanych jako jakie kategorie.

W drugim kroku wykonaliśmy to samo co w pierwszym, z tą różnicą, że wzieliśmy finalne dane z obrazkami komórek.

W trzecim kroku postanowiliśmy zrezygnować do trenowania z `train_test_split` na rzecz keras'owych datasetów. Jednak zostawiliśmy dane testowe i treningowe generowane przez funckję `train_test_split`, żeby móc potem wykorzystać je na modelu. Na końcu wyświetliliśmy wykres z kilkoma przykładowymi obrazkami z tesowego zbioru i kwalifikacją przyznaną im wraz z wartością procentową ukazującą jak dobrze został obrazek rozpoznany w danej kategorii. 

W czwartym etapie zmieniliśmy tylko model, żeby sprawdzić jak się zachowa przy użyciu innych warstw.

Poniżej znajduje się lista, z plikami z kodem oraz z danymi wykorzystanymi w danym pliku.

- ai-project-1.ipynb korzysta z data_1.zip
- ai-project-2.ipynb korzysta z data_2.zip
- ai-project-3.ipynb korzysta z data_2.zip
- ai-project-4.ipynb korzysta z data_2.zip

Do poszczególnych notebooków dodaliśmy komentarze nad każdą sekcją z informacją co się dzieje w danej części programu.
