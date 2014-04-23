# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from io import StringIO

__all__ = ['waltons_dataset', 'regression_dataset', 'lcd_dataset', 'dd_dataset']


def generate_left_censored_data():
    return {  
            'alluvial_fan': {
                    'T':[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 
                        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 
                        3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 7.0, 
                        7.0, 7.0, 8.0, 9.0],
                    'C':[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
            },
            'basin_trough': {
                    'T': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0,
                         3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 8.0, 9.0, 
                         9.0, 10.0, 10.0, 10.0, 10.0, 12.0, 14.0, 15.0, 15.0, 17.0, 23.0],
                    'C': [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 
                          0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1]
            }
    }


def generate_waltons_data():
    waltonG = np.array(['miR-137', 'miR-137', 'miR-137', 'miR-137', 'miR-137', 'miR-137',
                        'miR-137', 'miR-137', 'miR-137', 'miR-137', 'miR-137', 'miR-137',
                        'miR-137', 'miR-137', 'miR-137', 'miR-137', 'miR-137', 'miR-137',
                        'miR-137', 'miR-137', 'miR-137', 'miR-137', 'miR-137', 'miR-137',
                        'miR-137', 'miR-137', 'miR-137', 'miR-137', 'miR-137', 'miR-137',
                        'miR-137', 'miR-137', 'miR-137', 'miR-137', 'control', 'control',
                        'control', 'control', 'control', 'control', 'control', 'control',
                        'control', 'control', 'control', 'control', 'control', 'control',
                        'control', 'control', 'control', 'control', 'control', 'control',
                        'control', 'control', 'control', 'control', 'control', 'control',
                        'control', 'control', 'control', 'control', 'control', 'control',
                        'control', 'control', 'control', 'control', 'control', 'control',
                        'control', 'control', 'control', 'control', 'control', 'control',
                        'control', 'control', 'control', 'control', 'control', 'control',
                        'control', 'control', 'control', 'control', 'control', 'control',
                        'control', 'control', 'control', 'control', 'control', 'control',
                        'control', 'control', 'control', 'control', 'control', 'control',
                        'control', 'control', 'control', 'control', 'control', 'control',
                        'control', 'control', 'control', 'control', 'control', 'control',
                        'control', 'control', 'control', 'control', 'control', 'control',
                        'control', 'control', 'control', 'control', 'control', 'control',
                        'control', 'control', 'control', 'control', 'control', 'control',
                        'control', 'control', 'control', 'control', 'control', 'control',
                        'control', 'control', 'control', 'control', 'control', 'control',
                        'control', 'control', 'control', 'control', 'control', 'control',
                        'control', 'control', 'control', 'control', 'control', 'control',
                        'control', 'control', 'control', 'control', 'control', 'control',
                        'control'], dtype=object)

    waltonT = np.array([6., 13., 13., 13., 19., 19., 19., 26., 26., 26., 26.,
                        26., 33., 33., 47., 62., 62., 9., 9., 9., 15., 15.,
                        22., 22., 22., 22., 29., 29., 29., 29., 29., 36., 36.,
                        43., 33., 54., 54., 61., 61., 61., 61., 61., 61., 61.,
                        61., 61., 61., 61., 69., 69., 69., 69., 69., 69., 69.,
                        69., 69., 69., 69., 32., 53., 53., 60., 60., 60., 60.,
                        60., 68., 68., 68., 68., 68., 68., 68., 68., 68., 68.,
                        75., 17., 51., 51., 51., 58., 58., 58., 58., 66., 66.,
                        7., 7., 41., 41., 41., 41., 41., 41., 41., 48., 48.,
                        48., 48., 48., 48., 48., 48., 56., 56., 56., 56., 56.,
                        56., 56., 56., 56., 56., 56., 56., 56., 56., 56., 56.,
                        56., 56., 63., 63., 63., 63., 63., 63., 63., 63., 63.,
                        69., 69., 38., 38., 45., 45., 45., 45., 45., 45., 45.,
                        45., 45., 45., 53., 53., 53., 53., 53., 60., 60., 60.,
                        60., 60., 60., 60., 60., 60., 60., 60., 66.])

    waltonC = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1])

    waltons_data = pd.DataFrame(waltonT, columns=['T'])
    waltons_data['E'] = waltonC
    waltons_data['group'] = waltonG
    return waltons_data

def generate_dd_dataset():
    return pd.read_csv(
    StringIO(u"""
"id","ctryname","cowcode2","politycode","un_region_name","un_continent_name","ehead","leaderspellreg","democracy","regime","start_year","duration","observed"
"1","Afghanistan",700,700,"Southern Asia","Asia","Mohammad Zahir Shah","Mohammad Zahir Shah.Afghanistan.1946.1952.Monarchy","Non-democracy","Monarchy",1946,7,1
"2","Afghanistan",700,700,"Southern Asia","Asia","Sardar Mohammad Daoud","Sardar Mohammad Daoud.Afghanistan.1953.1962.Civilian Dict","Non-democracy","Civilian Dict",1953,10,1
"3","Afghanistan",700,700,"Southern Asia","Asia","Mohammad Zahir Shah","Mohammad Zahir Shah.Afghanistan.1963.1972.Monarchy","Non-democracy","Monarchy",1963,10,1
"4","Afghanistan",700,700,"Southern Asia","Asia","Sardar Mohammad Daoud","Sardar Mohammad Daoud.Afghanistan.1973.1977.Civilian Dict","Non-democracy","Civilian Dict",1973,5,0
"5","Afghanistan",700,700,"Southern Asia","Asia","Nur Mohammad Taraki","Nur Mohammad Taraki.Afghanistan.1978.1978.Civilian Dict","Non-democracy","Civilian Dict",1978,1,0
"6","Afghanistan",700,700,"Southern Asia","Asia","Babrak Karmal","Babrak Karmal.Afghanistan.1979.1984.Civilian Dict","Non-democracy","Civilian Dict",1979,6,1
"7","Afghanistan",700,700,"Southern Asia","Asia","Mohammed Najibullah","Mohammed Najibullah.Afghanistan.1985.1991.Civilian Dict","Non-democracy","Civilian Dict",1985,7,1
"8","Afghanistan",700,700,"Southern Asia","Asia","Burhanuddin Rabbani","Burhanuddin Rabbani.Afghanistan.1992.1995.Civilian Dict","Non-democracy","Civilian Dict",1992,4,1
"9","Afghanistan",700,700,"Southern Asia","Asia","Mullah Mohammad Rabbani","Mullah Mohammad Rabbani.Afghanistan.1996.2000.Civilian Dict","Non-democracy","Civilian Dict",1996,5,0
"10","Afghanistan",700,700,"Southern Asia","Asia","Hamid Karzai","Hamid Karzai.Afghanistan.2001.2008.Civilian Dict","Non-democracy","Civilian Dict",2001,8,0
"11","Albania",339,339,"Southern Europe","Europe","Enver Hoxha","Enver Hoxha.Albania.1946.1984.Military Dict","Non-democracy","Military Dict",1946,39,0
"12","Albania",339,339,"Southern Europe","Europe","Ramiz Alia","Ramiz Alia.Albania.1985.1990.Civilian Dict","Non-democracy","Civilian Dict",1985,6,1
"13","Albania",339,339,"Southern Europe","Europe","Ramiz Alia","Ramiz Alia.Albania.1991.1991.Parliamentary Dem","Democracy","Parliamentary Dem",1991,1,1
"14","Albania",339,339,"Southern Europe","Europe","Aleksander Meksi","Aleksander Meksi.Albania.1992.1996.Parliamentary Dem","Democracy","Parliamentary Dem",1992,5,1
"15","Albania",339,339,"Southern Europe","Europe","Fatos Nano","Fatos Nano.Albania.1997.1997.Parliamentary Dem","Democracy","Parliamentary Dem",1997,1,1
"16","Albania",339,339,"Southern Europe","Europe","Pandeli Majko","Pandeli Majko.Albania.1998.1998.Parliamentary Dem","Democracy","Parliamentary Dem",1998,1,1
"17","Albania",339,339,"Southern Europe","Europe","Ilir Meta","Ilir Meta.Albania.1999.2001.Parliamentary Dem","Democracy","Parliamentary Dem",1999,3,1
"18","Albania",339,339,"Southern Europe","Europe","Fatos Nano","Fatos Nano.Albania.2002.2004.Parliamentary Dem","Democracy","Parliamentary Dem",2002,3,1
"19","Albania",339,339,"Southern Europe","Europe","Sali Berisha","Sali Berisha.Albania.2005.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2005,4,0
"20","Algeria",615,615,"Northern Africa","Africa","Ahmed Ben Bella","Ahmed Ben Bella.Algeria.1962.1964.Civilian Dict","Non-democracy","Civilian Dict",1962,3,1
"21","Algeria",615,615,"Northern Africa","Africa","Houari Boumedienne","Houari Boumedienne.Algeria.1965.1977.Military Dict","Non-democracy","Military Dict",1965,13,0
"22","Algeria",615,615,"Northern Africa","Africa","military","military.Algeria.1978.1978.Military Dict","Non-democracy","Military Dict",1978,1,1
"23","Algeria",615,615,"Northern Africa","Africa","Benjedid Chadli","Benjedid Chadli.Algeria.1979.1991.Military Dict","Non-democracy","Military Dict",1979,13,1
"24","Algeria",615,615,"Northern Africa","Africa","military High Security Council","military High Security Council.Algeria.1992.1993.Military Dict","Non-democracy","Military Dict",1992,2,1
"25","Algeria",615,615,"Northern Africa","Africa","Liamine Zeroual","Liamine Zeroual.Algeria.1994.1998.Military Dict","Non-democracy","Military Dict",1994,5,1
"26","Algeria",615,615,"Northern Africa","Africa","Abdelaziz Bouteflika","Abdelaziz Bouteflika.Algeria.1999.2008.Civilian Dict","Non-democracy","Civilian Dict",1999,10,0
"27","Andorra",232,232,"Southern Europe","Europe","Oscar Ribas Reig","Oscar Ribas Reig.Andorra.1993.1993.Parliamentary Dem","Democracy","Parliamentary Dem",1993,1,1
"28","Andorra",232,232,"Southern Europe","Europe","Marc Forne Molne","Marc Forne Molne.Andorra.1994.2004.Parliamentary Dem","Democracy","Parliamentary Dem",1994,11,1
"29","Andorra",232,232,"Southern Europe","Europe","Albert Pintat Santolï¿½ria","Albert Pintat Santolï¿½ria.Andorra.2005.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2005,4,0
"30","Angola",540,540,"Middle Africa","Africa","Agostinho Neto","Agostinho Neto.Angola.1975.1978.Civilian Dict","Non-democracy","Civilian Dict",1975,4,0
"31","Angola",540,540,"Middle Africa","Africa","Jose Eduardo dos Santos","Jose Eduardo dos Santos.Angola.1979.2008.Civilian Dict","Non-democracy","Civilian Dict",1979,30,0
"32","Antigua & Barbuda",58,58,"Caribbean","Americas","Vere Cornwall Bird","Vere Cornwall Bird.Antigua & Barbuda.1981.1993.Parliamentary Dem","Democracy","Parliamentary Dem",1981,13,1
"33","Antigua & Barbuda",58,58,"Caribbean","Americas","Lester Bird","Lester Bird.Antigua & Barbuda.1994.2003.Parliamentary Dem","Democracy","Parliamentary Dem",1994,10,1
"34","Antigua & Barbuda",58,58,"Caribbean","Americas","Baldwin Spencer","Baldwin Spencer.Antigua & Barbuda.2004.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2004,5,0
"35","Argentina",160,160,"South America","Americas","Juan Peron","Juan Peron.Argentina.1946.1954.Presidential Dem","Democracy","Presidential Dem",1946,9,1
"36","Argentina",160,160,"South America","Americas","Pedro Eugenio Aramburu Cilveti","Pedro Eugenio Aramburu Cilveti.Argentina.1955.1957.Military Dict","Non-democracy","Military Dict",1955,3,1
"37","Argentina",160,160,"South America","Americas","Arturo Frondizi","Arturo Frondizi.Argentina.1958.1961.Presidential Dem","Democracy","Presidential Dem",1958,4,1
"38","Argentina",160,160,"South America","Americas","Military","Military.Argentina.1962.1962.Military Dict","Non-democracy","Military Dict",1962,1,1
"39","Argentina",160,160,"South America","Americas","Arturo Umberto Illia","Arturo Umberto Illia.Argentina.1963.1965.Presidential Dem","Democracy","Presidential Dem",1963,3,1
"40","Argentina",160,160,"South America","Americas","Juan Carlos Ongania","Juan Carlos Ongania.Argentina.1966.1969.Military Dict","Non-democracy","Military Dict",1966,4,1
"41","Argentina",160,160,"South America","Americas","Roberto Marcelo Levingston","Roberto Marcelo Levingston.Argentina.1970.1970.Military Dict","Non-democracy","Military Dict",1970,1,1
"42","Argentina",160,160,"South America","Americas","Alejandro Agustan Lanusse","Alejandro Agustan Lanusse.Argentina.1971.1972.Military Dict","Non-democracy","Military Dict",1971,2,1
"43","Argentina",160,160,"South America","Americas","Juan Peron","Juan Peron.Argentina.1973.1973.Presidential Dem","Democracy","Presidential Dem",1973,1,0
"44","Argentina",160,160,"South America","Americas","Isabel Peron","Isabel Peron.Argentina.1974.1975.Presidential Dem","Democracy","Presidential Dem",1974,2,1
"45","Argentina",160,160,"South America","Americas","Jorge Rafael Videla","Jorge Rafael Videla.Argentina.1976.1980.Military Dict","Non-democracy","Military Dict",1976,5,1
"46","Argentina",160,160,"South America","Americas","Leopoldo Galtieri","Leopoldo Galtieri.Argentina.1981.1981.Military Dict","Non-democracy","Military Dict",1981,1,1
"47","Argentina",160,160,"South America","Americas","Reynaldo Bignone","Reynaldo Bignone.Argentina.1982.1982.Military Dict","Non-democracy","Military Dict",1982,1,1
"48","Argentina",160,160,"South America","Americas","Raul Alfonsin","Raul Alfonsin.Argentina.1983.1988.Presidential Dem","Democracy","Presidential Dem",1983,6,1
"49","Argentina",160,160,"South America","Americas","Carlos Menem","Carlos Menem.Argentina.1989.1998.Presidential Dem","Democracy","Presidential Dem",1989,10,1
"50","Argentina",160,160,"South America","Americas","Fernando de la Rua","Fernando de la Rua.Argentina.1999.2000.Presidential Dem","Democracy","Presidential Dem",1999,2,1
"51","Argentina",160,160,"South America","Americas","Eduardo Oscar Camano","Eduardo Oscar Camano.Argentina.2001.2001.Presidential Dem","Democracy","Presidential Dem",2001,1,1
"52","Argentina",160,160,"South America","Americas","Eduardo Alberto Duhalde Maldonado","Eduardo Alberto Duhalde Maldonado.Argentina.2002.2004.Presidential Dem","Democracy","Presidential Dem",2002,3,1
"53","Argentina",160,160,"South America","Americas","Nï¿½stor Carlos Kirchner Ostoic","Nï¿½stor Carlos Kirchner Ostoic.Argentina.2005.2006.Presidential Dem","Democracy","Presidential Dem",2005,2,1
"54","Argentina",160,160,"South America","Americas","Cristina Fernï¿½ndez de Kirchner","Cristina Fernï¿½ndez de Kirchner.Argentina.2007.2008.Presidential Dem","Democracy","Presidential Dem",2007,2,0
"55","Armenia",371,371,"Western Asia","Asia","Levon Ter-Petrosyan","Levon Ter-Petrosyan.Armenia.1991.1994.Mixed Dem","Democracy","Mixed Dem",1991,4,1
"56","Armenia",371,371,"Western Asia","Asia","Hrant Bagratyan","Hrant Bagratyan.Armenia.1995.1995.Mixed Dem","Democracy","Mixed Dem",1995,1,1
"57","Armenia",371,371,"Western Asia","Asia","Armen Sarkisyan","Armen Sarkisyan.Armenia.1996.1996.Mixed Dem","Democracy","Mixed Dem",1996,1,1
"58","Armenia",371,371,"Western Asia","Asia","Robert Kocharyan","Robert Kocharyan.Armenia.1997.1997.Mixed Dem","Democracy","Mixed Dem",1997,1,1
"59","Armenia",371,371,"Western Asia","Asia","Armen Darbinyan","Armen Darbinyan.Armenia.1998.1998.Mixed Dem","Democracy","Mixed Dem",1998,1,1
"60","Armenia",371,371,"Western Asia","Asia","Aram Sargsyan","Aram Sargsyan.Armenia.1999.1999.Mixed Dem","Democracy","Mixed Dem",1999,1,1
"61","Armenia",371,371,"Western Asia","Asia","Andranik Margaryan","Andranik Margaryan.Armenia.2000.2006.Mixed Dem","Democracy","Mixed Dem",2000,7,0
"62","Armenia",371,371,"Western Asia","Asia","Serzh Sargsyan","Serzh Sargsyan.Armenia.2007.2007.Mixed Dem","Democracy","Mixed Dem",2007,1,1
"63","Armenia",371,371,"Western Asia","Asia","Tigran Sargsyan","Tigran Sargsyan.Armenia.2008.2008.Mixed Dem","Democracy","Mixed Dem",2008,1,0
"64","Australia",900,900,"Australia and New Zealand","Oceania","Joseph Chifley","Joseph Chifley.Australia.1946.1948.Parliamentary Dem","Democracy","Parliamentary Dem",1946,3,1
"65","Australia",900,900,"Australia and New Zealand","Oceania","Robert Menzies","Robert Menzies.Australia.1949.1965.Parliamentary Dem","Democracy","Parliamentary Dem",1949,17,1
"66","Australia",900,900,"Australia and New Zealand","Oceania","Harold Holt","Harold Holt.Australia.1966.1966.Parliamentary Dem","Democracy","Parliamentary Dem",1966,1,0
"67","Australia",900,900,"Australia and New Zealand","Oceania","John McEwan","John McEwan.Australia.1967.1967.Parliamentary Dem","Democracy","Parliamentary Dem",1967,1,1
"68","Australia",900,900,"Australia and New Zealand","Oceania","John Gorton","John Gorton.Australia.1968.1970.Parliamentary Dem","Democracy","Parliamentary Dem",1968,3,1
"69","Australia",900,900,"Australia and New Zealand","Oceania","William McMahon","William McMahon.Australia.1971.1971.Parliamentary Dem","Democracy","Parliamentary Dem",1971,1,1
"70","Australia",900,900,"Australia and New Zealand","Oceania","E. Gough Whitlam","E. Gough Whitlam.Australia.1972.1974.Parliamentary Dem","Democracy","Parliamentary Dem",1972,3,1
"71","Australia",900,900,"Australia and New Zealand","Oceania","Malcom Fraser","Malcom Fraser.Australia.1975.1982.Parliamentary Dem","Democracy","Parliamentary Dem",1975,8,1
"72","Australia",900,900,"Australia and New Zealand","Oceania","Robert Hawke","Robert Hawke.Australia.1983.1990.Parliamentary Dem","Democracy","Parliamentary Dem",1983,8,1
"73","Australia",900,900,"Australia and New Zealand","Oceania","Paul Keating","Paul Keating.Australia.1991.1995.Parliamentary Dem","Democracy","Parliamentary Dem",1991,5,1
"74","Australia",900,900,"Australia and New Zealand","Oceania","John Howard","John Howard.Australia.1996.2006.Parliamentary Dem","Democracy","Parliamentary Dem",1996,11,1
"75","Australia",900,900,"Australia and New Zealand","Oceania","Kevin Michael Rudd","Kevin Michael Rudd.Australia.2007.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2007,2,0
"76","Austria",305,305,"Western Europe","Europe","Leopold Figl","Leopold Figl.Austria.1946.1952.Mixed Dem","Democracy","Mixed Dem",1946,7,1
"77","Austria",305,305,"Western Europe","Europe","Julius Raab","Julius Raab.Austria.1953.1960.Mixed Dem","Democracy","Mixed Dem",1953,8,1
"78","Austria",305,305,"Western Europe","Europe","Alfons Gorbach","Alfons Gorbach.Austria.1961.1963.Mixed Dem","Democracy","Mixed Dem",1961,3,1
"79","Austria",305,305,"Western Europe","Europe","Josef Klaus","Josef Klaus.Austria.1964.1969.Mixed Dem","Democracy","Mixed Dem",1964,6,1
"80","Austria",305,305,"Western Europe","Europe","Bruno Kreisky","Bruno Kreisky.Austria.1970.1982.Mixed Dem","Democracy","Mixed Dem",1970,13,1
"81","Austria",305,305,"Western Europe","Europe","Alfred Sinowatz","Alfred Sinowatz.Austria.1983.1985.Mixed Dem","Democracy","Mixed Dem",1983,3,1
"82","Austria",305,305,"Western Europe","Europe","Franz Vranitzky","Franz Vranitzky.Austria.1986.1996.Mixed Dem","Democracy","Mixed Dem",1986,11,1
"83","Austria",305,305,"Western Europe","Europe","Viktor Klima","Viktor Klima.Austria.1997.1999.Mixed Dem","Democracy","Mixed Dem",1997,3,1
"84","Austria",305,305,"Western Europe","Europe","Wolfgang Schussel","Wolfgang Schussel.Austria.2000.2006.Mixed Dem","Democracy","Mixed Dem",2000,7,1
"85","Austria",305,305,"Western Europe","Europe","Alfred Gusenbauer","Alfred Gusenbauer.Austria.2007.2008.Mixed Dem","Democracy","Mixed Dem",2007,2,0
"86","Azerbaijan",373,373,"Western Asia","Asia","Ayaz Mutalibov","Ayaz Mutalibov.Azerbaijan.1991.1991.Civilian Dict","Non-democracy","Civilian Dict",1991,1,1
"87","Azerbaijan",373,373,"Western Asia","Asia","Abulfez Elchibey","Abulfez Elchibey.Azerbaijan.1992.1992.Civilian Dict","Non-democracy","Civilian Dict",1992,1,1
"88","Azerbaijan",373,373,"Western Asia","Asia","Heydar Aliyev","Heydar Aliyev.Azerbaijan.1993.2002.Civilian Dict","Non-democracy","Civilian Dict",1993,10,0
"89","Azerbaijan",373,373,"Western Asia","Asia","Ilham Heydar oglu Aliyev","Ilham Heydar oglu Aliyev.Azerbaijan.2003.2008.Civilian Dict","Non-democracy","Civilian Dict",2003,6,0
"90","Bahamas",31,31,"Caribbean","Americas","Lynden Pindling","Lynden Pindling.Bahamas.1973.1991.Parliamentary Dem","Democracy","Parliamentary Dem",1973,19,1
"91","Bahamas",31,31,"Caribbean","Americas","Hubert Ingraham","Hubert Ingraham.Bahamas.1992.2001.Parliamentary Dem","Democracy","Parliamentary Dem",1992,10,1
"92","Bahamas",31,31,"Caribbean","Americas","Perry Gladstone Christie","Perry Gladstone Christie.Bahamas.2002.2006.Parliamentary Dem","Democracy","Parliamentary Dem",2002,5,1
"93","Bahamas",31,31,"Caribbean","Americas","Hubert Alexander Ingraham","Hubert Alexander Ingraham.Bahamas.2007.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2007,2,0
"94","Bahrain",692,692,"Western Asia","Asia","Sheikh 'Isa ibn Sulman Al Khalifah","Sheikh 'Isa ibn Sulman Al Khalifah.Bahrain.1971.1998.Monarchy","Non-democracy","Monarchy",1971,28,0
"95","Bahrain",692,692,"Western Asia","Asia","Sheikh Hamad ibn 'Isa Al Khalifah","Sheikh Hamad ibn 'Isa Al Khalifah.Bahrain.1999.2008.Monarchy","Non-democracy","Monarchy",1999,10,0
"96","Bangladesh",771,771,"Southern Asia","Asia","Tajuddin Ahmed","Tajuddin Ahmed.Bangladesh.1971.1971.Civilian Dict","Non-democracy","Civilian Dict",1971,1,1
"97","Bangladesh",771,771,"Southern Asia","Asia","Mujibur Rahman","Mujibur Rahman.Bangladesh.1972.1974.Civilian Dict","Non-democracy","Civilian Dict",1972,3,0
"98","Bangladesh",771,771,"Southern Asia","Asia","Abu Sadat Mohammad Sayem","Abu Sadat Mohammad Sayem.Bangladesh.1975.1976.Civilian Dict","Non-democracy","Civilian Dict",1975,2,1
"99","Bangladesh",771,771,"Southern Asia","Asia","Zia ur-Rahman","Zia ur-Rahman.Bangladesh.1977.1980.Military Dict","Non-democracy","Military Dict",1977,4,0
"100","Bangladesh",771,771,"Southern Asia","Asia","Abdus Sattar","Abdus Sattar.Bangladesh.1981.1981.Civilian Dict","Non-democracy","Civilian Dict",1981,1,1
"101","Bangladesh",771,771,"Southern Asia","Asia","Hossain Mohammad Ershad","Hossain Mohammad Ershad.Bangladesh.1982.1985.Military Dict","Non-democracy","Military Dict",1982,4,1
"102","Bangladesh",771,771,"Southern Asia","Asia","Hossain Mohammad Ershad","Hossain Mohammad Ershad.Bangladesh.1986.1989.Mixed Dem","Democracy","Mixed Dem",1986,4,1
"103","Bangladesh",771,771,"Southern Asia","Asia","Shahabuddin Ahmed","Shahabuddin Ahmed.Bangladesh.1990.1990.Mixed Dem","Democracy","Mixed Dem",1990,1,1
"104","Bangladesh",771,771,"Southern Asia","Asia","Khaleda Zia","Khaleda Zia.Bangladesh.1991.1995.Parliamentary Dem","Democracy","Parliamentary Dem",1991,5,1
"105","Bangladesh",771,771,"Southern Asia","Asia","Sheikh Hasina Wajed","Sheikh Hasina Wajed.Bangladesh.1996.2000.Parliamentary Dem","Democracy","Parliamentary Dem",1996,5,1
"106","Bangladesh",771,771,"Southern Asia","Asia","Khaleda Zia","Khaleda Zia.Bangladesh.2001.2006.Parliamentary Dem","Democracy","Parliamentary Dem",2001,6,1
"107","Bangladesh",771,771,"Southern Asia","Asia","military","military.Bangladesh.2007.2007.Military Dict","Non-democracy","Military Dict",2007,1,1
"108","Bangladesh",771,771,"Southern Asia","Asia","military","military.Bangladesh.2008.2008.Civilian Dict","Non-democracy","Civilian Dict",2008,1,0
"109","Barbados",53,53,"Caribbean","Americas","Errol Barrow","Errol Barrow.Barbados.1966.1975.Parliamentary Dem","Democracy","Parliamentary Dem",1966,10,1
"110","Barbados",53,53,"Caribbean","Americas","John Adams","John Adams.Barbados.1976.1984.Parliamentary Dem","Democracy","Parliamentary Dem",1976,9,0
"111","Barbados",53,53,"Caribbean","Americas","Bernard St. John","Bernard St. John.Barbados.1985.1985.Parliamentary Dem","Democracy","Parliamentary Dem",1985,1,1
"112","Barbados",53,53,"Caribbean","Americas","Errol Barrow","Errol Barrow.Barbados.1986.1986.Parliamentary Dem","Democracy","Parliamentary Dem",1986,1,0
"113","Barbados",53,53,"Caribbean","Americas","Erskine Sandiford","Erskine Sandiford.Barbados.1987.1993.Parliamentary Dem","Democracy","Parliamentary Dem",1987,7,1
"114","Barbados",53,53,"Caribbean","Americas","Owen Arthur","Owen Arthur.Barbados.1994.2007.Parliamentary Dem","Democracy","Parliamentary Dem",1994,14,1
"115","Barbados",53,53,"Caribbean","Americas","David Thompson","David Thompson.Barbados.2008.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2008,1,0
"116","Belarus",370,370,"Eastern Europe","Europe","Stanislau Shushkevich","Stanislau Shushkevich.Belarus.1991.1993.Civilian Dict","Non-democracy","Civilian Dict",1991,3,1
"117","Belarus",370,370,"Eastern Europe","Europe","Alyaksandr Lukashenka","Alyaksandr Lukashenka.Belarus.1994.2008.Civilian Dict","Non-democracy","Civilian Dict",1994,15,0
"118","Belgium",211,211,"Western Europe","Europe","Camille Huysmans","Camille Huysmans.Belgium.1946.1946.Parliamentary Dem","Democracy","Parliamentary Dem",1946,1,1
"119","Belgium",211,211,"Western Europe","Europe","Paul-Henri Spaak","Paul-Henri Spaak.Belgium.1947.1948.Parliamentary Dem","Democracy","Parliamentary Dem",1947,2,1
"120","Belgium",211,211,"Western Europe","Europe","Gaston Eyskens","Gaston Eyskens.Belgium.1949.1949.Parliamentary Dem","Democracy","Parliamentary Dem",1949,1,1
"121","Belgium",211,211,"Western Europe","Europe","Joseph Pholien","Joseph Pholien.Belgium.1950.1951.Parliamentary Dem","Democracy","Parliamentary Dem",1950,2,1
"122","Belgium",211,211,"Western Europe","Europe","Jean van Houtte","Jean van Houtte.Belgium.1952.1953.Parliamentary Dem","Democracy","Parliamentary Dem",1952,2,1
"123","Belgium",211,211,"Western Europe","Europe","Achille van Acker","Achille van Acker.Belgium.1954.1957.Parliamentary Dem","Democracy","Parliamentary Dem",1954,4,1
"124","Belgium",211,211,"Western Europe","Europe","Gaston Eyskens","Gaston Eyskens.Belgium.1958.1960.Parliamentary Dem","Democracy","Parliamentary Dem",1958,3,1
"125","Belgium",211,211,"Western Europe","Europe","Theodore Lefevre","Theodore Lefevre.Belgium.1961.1964.Parliamentary Dem","Democracy","Parliamentary Dem",1961,4,1
"126","Belgium",211,211,"Western Europe","Europe","Pierre Harmel","Pierre Harmel.Belgium.1965.1965.Parliamentary Dem","Democracy","Parliamentary Dem",1965,1,1
"127","Belgium",211,211,"Western Europe","Europe","Paul van den Boeynants","Paul van den Boeynants.Belgium.1966.1967.Parliamentary Dem","Democracy","Parliamentary Dem",1966,2,1
"128","Belgium",211,211,"Western Europe","Europe","Gaston Eyskens","Gaston Eyskens.Belgium.1968.1972.Parliamentary Dem","Democracy","Parliamentary Dem",1968,5,1
"129","Belgium",211,211,"Western Europe","Europe","Edmond Leburton","Edmond Leburton.Belgium.1973.1973.Parliamentary Dem","Democracy","Parliamentary Dem",1973,1,1
"130","Belgium",211,211,"Western Europe","Europe","Leo Tindemans","Leo Tindemans.Belgium.1974.1977.Parliamentary Dem","Democracy","Parliamentary Dem",1974,4,1
"131","Belgium",211,211,"Western Europe","Europe","Paul van den Boeynants","Paul van den Boeynants.Belgium.1978.1978.Parliamentary Dem","Democracy","Parliamentary Dem",1978,1,1
"132","Belgium",211,211,"Western Europe","Europe","Wilfried Martens","Wilfried Martens.Belgium.1979.1991.Parliamentary Dem","Democracy","Parliamentary Dem",1979,13,1
"133","Belgium",211,211,"Western Europe","Europe","Jean-Luc Dehaene","Jean-Luc Dehaene.Belgium.1992.1998.Parliamentary Dem","Democracy","Parliamentary Dem",1992,7,1
"134","Belgium",211,211,"Western Europe","Europe","Guy Verhofstadt","Guy Verhofstadt.Belgium.1999.2007.Parliamentary Dem","Democracy","Parliamentary Dem",1999,9,1
"135","Belgium",211,211,"Western Europe","Europe","Herman Van Rompuy","Herman Van Rompuy.Belgium.2008.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2008,1,0
"136","Belize",80,80,"Central America","Americas","George Cadle Price","George Cadle Price.Belize.1981.1983.Parliamentary Dem","Democracy","Parliamentary Dem",1981,3,1
"137","Belize",80,80,"Central America","Americas","Manuel Esquivel","Manuel Esquivel.Belize.1984.1988.Parliamentary Dem","Democracy","Parliamentary Dem",1984,5,1
"138","Belize",80,80,"Central America","Americas","George Cadle Price","George Cadle Price.Belize.1989.1992.Parliamentary Dem","Democracy","Parliamentary Dem",1989,4,1
"139","Belize",80,80,"Central America","Americas","Manuel Esquivel","Manuel Esquivel.Belize.1993.1997.Parliamentary Dem","Democracy","Parliamentary Dem",1993,5,1
"140","Belize",80,80,"Central America","Americas","Said Musa","Said Musa.Belize.1998.2007.Parliamentary Dem","Democracy","Parliamentary Dem",1998,10,1
"141","Belize",80,80,"Central America","Americas","Dean Oliver Barrow","Dean Oliver Barrow.Belize.2008.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2008,1,0
"142","Benin",434,434,"Western Africa","Africa","Hubert Maga","Hubert Maga.Benin.1960.1962.Civilian Dict","Non-democracy","Civilian Dict",1960,3,1
"143","Benin",434,434,"Western Africa","Africa","Christophe Soglo","Christophe Soglo.Benin.1963.1966.Military Dict","Non-democracy","Military Dict",1963,4,1
"144","Benin",434,434,"Western Africa","Africa","Alphonse Alley","Alphonse Alley.Benin.1967.1967.Military Dict","Non-democracy","Military Dict",1967,1,1
"145","Benin",434,434,"Western Africa","Africa","military","military.Benin.1968.1969.Military Dict","Non-democracy","Military Dict",1968,2,1
"146","Benin",434,434,"Western Africa","Africa","Hubert Maga","Hubert Maga.Benin.1970.1971.Civilian Dict","Non-democracy","Civilian Dict",1970,2,1
"147","Benin",434,434,"Western Africa","Africa","Mathieu Kerekou","Mathieu Kerekou.Benin.1972.1990.Military Dict","Non-democracy","Military Dict",1972,19,1
"148","Benin",434,434,"Western Africa","Africa","Nicephore Soglo","Nicephore Soglo.Benin.1991.1995.Presidential Dem","Democracy","Presidential Dem",1991,5,1
"149","Benin",434,434,"Western Africa","Africa","Mathieu Kerekou","Mathieu Kerekou.Benin.1996.2005.Presidential Dem","Democracy","Presidential Dem",1996,10,1
"150","Benin",434,434,"Western Africa","Africa","Yayi Boni","Yayi Boni.Benin.2006.2008.Presidential Dem","Democracy","Presidential Dem",2006,3,0
"151","Bhutan",760,760,"Southern Asia","Asia","Jigme Wangchuk","Jigme Wangchuk.Bhutan.1946.1951.Monarchy","Non-democracy","Monarchy",1946,6,0
"152","Bhutan",760,760,"Southern Asia","Asia","Jigme Dorji Wangchuk","Jigme Dorji Wangchuk.Bhutan.1952.1971.Monarchy","Non-democracy","Monarchy",1952,20,0
"153","Bhutan",760,760,"Southern Asia","Asia","Jigme Singye Wangchuk","Jigme Singye Wangchuk.Bhutan.1972.2006.Monarchy","Non-democracy","Monarchy",1972,35,1
"154","Bhutan",760,760,"Southern Asia","Asia","Kinzang Dorji","Kinzang Dorji.Bhutan.2007.2007.Parliamentary Dem","Democracy","Parliamentary Dem",2007,1,1
"155","Bhutan",760,760,"Southern Asia","Asia","Jigme Thinley","Jigme Thinley.Bhutan.2008.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2008,1,0
"156","Bolivia",145,145,"South America","Americas","Military","Military.Bolivia.1946.1946.Military Dict","Non-democracy","Military Dict",1946,1,1
"157","Bolivia",145,145,"South America","Americas","Enrique Hertzog","Enrique Hertzog.Bolivia.1947.1948.Civilian Dict","Non-democracy","Civilian Dict",1947,2,1
"158","Bolivia",145,145,"South America","Americas","Mamerto Urriolagoitia","Mamerto Urriolagoitia.Bolivia.1949.1950.Civilian Dict","Non-democracy","Civilian Dict",1949,2,1
"159","Bolivia",145,145,"South America","Americas","Hugo Ballivion Rojas","Hugo Ballivion Rojas.Bolivia.1951.1951.Military Dict","Non-democracy","Military Dict",1951,1,1
"160","Bolivia",145,145,"South America","Americas","Victor Paz Estenssoro","Victor Paz Estenssoro.Bolivia.1952.1955.Civilian Dict","Non-democracy","Civilian Dict",1952,4,1
"161","Bolivia",145,145,"South America","Americas","Hernan Siles Zuazo","Hernan Siles Zuazo.Bolivia.1956.1959.Civilian Dict","Non-democracy","Civilian Dict",1956,4,1
"162","Bolivia",145,145,"South America","Americas","Victor Paz Estenssoro","Victor Paz Estenssoro.Bolivia.1960.1963.Civilian Dict","Non-democracy","Civilian Dict",1960,4,1
"163","Bolivia",145,145,"South America","Americas","Rene Barrientos Ortuno","Rene Barrientos Ortuno.Bolivia.1964.1964.Military Dict","Non-democracy","Military Dict",1964,1,1
"164","Bolivia",145,145,"South America","Americas","Rene Barrientos Ortuno+Alfredo Ovando Candia","Rene Barrientos Ortuno+Alfredo Ovando Candia.Bolivia.1965.1965.Military Dict","Non-democracy","Military Dict",1965,1,1
"165","Bolivia",145,145,"South America","Americas","Rene Barrientos Ortuno","Rene Barrientos Ortuno.Bolivia.1966.1968.Military Dict","Non-democracy","Military Dict",1966,3,0
"166","Bolivia",145,145,"South America","Americas","Alfredo Ovando Candia","Alfredo Ovando Candia.Bolivia.1969.1969.Military Dict","Non-democracy","Military Dict",1969,1,1
"167","Bolivia",145,145,"South America","Americas","Juan Torres Gonzalez","Juan Torres Gonzalez.Bolivia.1970.1970.Military Dict","Non-democracy","Military Dict",1970,1,1
"168","Bolivia",145,145,"South America","Americas","Hugo Banzer Suarez","Hugo Banzer Suarez.Bolivia.1971.1977.Military Dict","Non-democracy","Military Dict",1971,7,1
"169","Bolivia",145,145,"South America","Americas","David Padilla Arancibia","David Padilla Arancibia.Bolivia.1978.1978.Military Dict","Non-democracy","Military Dict",1978,1,1
"170","Bolivia",145,145,"South America","Americas","Lidia Gueiler Tejada","Lidia Gueiler Tejada.Bolivia.1979.1979.Presidential Dem","Democracy","Presidential Dem",1979,1,1
"171","Bolivia",145,145,"South America","Americas","Luis Garcia Meza Tejada","Luis Garcia Meza Tejada.Bolivia.1980.1980.Military Dict","Non-democracy","Military Dict",1980,1,1
"172","Bolivia",145,145,"South America","Americas","Celso Torrelio Villa","Celso Torrelio Villa.Bolivia.1981.1981.Military Dict","Non-democracy","Military Dict",1981,1,1
"173","Bolivia",145,145,"South America","Americas","Hernan Siles Zuazo","Hernan Siles Zuazo.Bolivia.1982.1984.Presidential Dem","Democracy","Presidential Dem",1982,3,1
"174","Bolivia",145,145,"South America","Americas","Victor Paz Estenssoro","Victor Paz Estenssoro.Bolivia.1985.1988.Presidential Dem","Democracy","Presidential Dem",1985,4,1
"175","Bolivia",145,145,"South America","Americas","Jaime Paz Zamoro","Jaime Paz Zamoro.Bolivia.1989.1992.Presidential Dem","Democracy","Presidential Dem",1989,4,1
"176","Bolivia",145,145,"South America","Americas","Gonzalo Sanchez de Lozada","Gonzalo Sanchez de Lozada.Bolivia.1993.1996.Presidential Dem","Democracy","Presidential Dem",1993,4,1
"177","Bolivia",145,145,"South America","Americas","Hugo Banzer Suarez","Hugo Banzer Suarez.Bolivia.1997.2000.Presidential Dem","Democracy","Presidential Dem",1997,4,1
"178","Bolivia",145,145,"South America","Americas","Jorge Fernando Quiroga Ramirez","Jorge Fernando Quiroga Ramirez.Bolivia.2001.2001.Presidential Dem","Democracy","Presidential Dem",2001,1,1
"179","Bolivia",145,145,"South America","Americas","Gonzalo Sanchez de Lozada","Gonzalo Sanchez de Lozada.Bolivia.2002.2002.Presidential Dem","Democracy","Presidential Dem",2002,1,1
"180","Bolivia",145,145,"South America","Americas","Carlos Diego Mesa Gisbert","Carlos Diego Mesa Gisbert.Bolivia.2003.2004.Presidential Dem","Democracy","Presidential Dem",2003,2,1
"181","Bolivia",145,145,"South America","Americas","Eduardo Rodrï¿½guez Veltzï¿½","Eduardo Rodrï¿½guez Veltzï¿½.Bolivia.2005.2005.Presidential Dem","Democracy","Presidential Dem",2005,1,1
"182","Bolivia",145,145,"South America","Americas","Juan Evo Morales Ayma","Juan Evo Morales Ayma.Bolivia.2006.2008.Presidential Dem","Democracy","Presidential Dem",2006,3,0
"183","Bosnia and Herzegovina",346,346,"Southern Europe","Europe","Alija Izetbegovic","Alija Izetbegovic.Bosnia and Herzegovina.1991.1997.Civilian Dict","Non-democracy","Civilian Dict",1991,7,1
"184","Bosnia and Herzegovina",346,346,"Southern Europe","Europe","Zivko Radisic","Zivko Radisic.Bosnia and Herzegovina.1998.1998.Civilian Dict","Non-democracy","Civilian Dict",1998,1,1
"185","Bosnia and Herzegovina",346,346,"Southern Europe","Europe","Ante Jelavic","Ante Jelavic.Bosnia and Herzegovina.1999.1999.Civilian Dict","Non-democracy","Civilian Dict",1999,1,1
"186","Bosnia and Herzegovina",346,346,"Southern Europe","Europe","Zivko Radisic","Zivko Radisic.Bosnia and Herzegovina.2000.2000.Civilian Dict","Non-democracy","Civilian Dict",2000,1,1
"187","Bosnia and Herzegovina",346,346,"Southern Europe","Europe","Jozo Krizanovic","Jozo Krizanovic.Bosnia and Herzegovina.2001.2001.Civilian Dict","Non-democracy","Civilian Dict",2001,1,1
"188","Bosnia and Herzegovina",346,346,"Southern Europe","Europe","Mirko Sarovic","Mirko Sarovic.Bosnia and Herzegovina.2002.2002.Civilian Dict","Non-democracy","Civilian Dict",2002,1,1
"189","Bosnia and Herzegovina",346,346,"Southern Europe","Europe","Dragan Kovic","Dragan Kovic.Bosnia and Herzegovina.2003.2003.Civilian Dict","Non-democracy","Civilian Dict",2003,1,1
"190","Bosnia and Herzegovina",346,346,"Southern Europe","Europe","Borislav Paravac","Borislav Paravac.Bosnia and Herzegovina.2004.2004.Civilian Dict","Non-democracy","Civilian Dict",2004,1,1
"191","Bosnia and Herzegovina",346,346,"Southern Europe","Europe","Ivo Miro Jovic","Ivo Miro Jovic.Bosnia and Herzegovina.2005.2005.Civilian Dict","Non-democracy","Civilian Dict",2005,1,1
"192","Bosnia and Herzegovina",346,346,"Southern Europe","Europe","Nebojsa Radmanovic","Nebojsa Radmanovic.Bosnia and Herzegovina.2006.2006.Civilian Dict","Non-democracy","Civilian Dict",2006,1,1
"193","Bosnia and Herzegovina",346,346,"Southern Europe","Europe","Zeljko Komsic","Zeljko Komsic.Bosnia and Herzegovina.2007.2007.Civilian Dict","Non-democracy","Civilian Dict",2007,1,1
"194","Bosnia and Herzegovina",346,346,"Southern Europe","Europe","Nebojsa Radmanovic","Nebojsa Radmanovic.Bosnia and Herzegovina.2008.2008.Civilian Dict","Non-democracy","Civilian Dict",2008,1,0
"195","Botswana",571,571,"Southern Africa","Africa","Seretse Khama","Seretse Khama.Botswana.1966.1979.Civilian Dict","Non-democracy","Civilian Dict",1966,14,0
"196","Botswana",571,571,"Southern Africa","Africa","Quett Masire","Quett Masire.Botswana.1980.1997.Civilian Dict","Non-democracy","Civilian Dict",1980,18,1
"197","Botswana",571,571,"Southern Africa","Africa","Festus Mogae","Festus Mogae.Botswana.1998.2007.Civilian Dict","Non-democracy","Civilian Dict",1998,10,1
"198","Botswana",571,571,"Southern Africa","Africa","Seretse Khama Ian Khama","Seretse Khama Ian Khama.Botswana.2008.2008.Military Dict","Non-democracy","Military Dict",2008,1,0
"199","Brazil",140,140,"South America","Americas","Eurico Dutra","Eurico Dutra.Brazil.1946.1950.Presidential Dem","Democracy","Presidential Dem",1946,5,1
"200","Brazil",140,140,"South America","Americas","Getulio Vargas","Getulio Vargas.Brazil.1951.1953.Presidential Dem","Democracy","Presidential Dem",1951,3,0
"201","Brazil",140,140,"South America","Americas","Joao Cafe","Joao Cafe.Brazil.1954.1954.Presidential Dem","Democracy","Presidential Dem",1954,1,1
"202","Brazil",140,140,"South America","Americas","Nereu Ramos","Nereu Ramos.Brazil.1955.1955.Presidential Dem","Democracy","Presidential Dem",1955,1,1
"203","Brazil",140,140,"South America","Americas","Juscelino Kubitschek","Juscelino Kubitschek.Brazil.1956.1960.Presidential Dem","Democracy","Presidential Dem",1956,5,1
"204","Brazil",140,140,"South America","Americas","Tancredo de Almeida Neves","Tancredo de Almeida Neves.Brazil.1961.1961.Mixed Dem","Democracy","Mixed Dem",1961,1,1
"205","Brazil",140,140,"South America","Americas","Hermes Lima","Hermes Lima.Brazil.1962.1962.Mixed Dem","Democracy","Mixed Dem",1962,1,1
"206","Brazil",140,140,"South America","Americas","Joao Belchior Marques Goulart","Joao Belchior Marques Goulart.Brazil.1963.1963.Presidential Dem","Democracy","Presidential Dem",1963,1,1
"207","Brazil",140,140,"South America","Americas","Humberto de Alencar Castello Branco","Humberto de Alencar Castello Branco.Brazil.1964.1966.Military Dict","Non-democracy","Military Dict",1964,3,1
"208","Brazil",140,140,"South America","Americas","Artur da Costa e Silva","Artur da Costa e Silva.Brazil.1967.1968.Military Dict","Non-democracy","Military Dict",1967,2,1
"209","Brazil",140,140,"South America","Americas","Emilio Medici","Emilio Medici.Brazil.1969.1973.Military Dict","Non-democracy","Military Dict",1969,5,1
"210","Brazil",140,140,"South America","Americas","Ernesto Geisel","Ernesto Geisel.Brazil.1974.1978.Military Dict","Non-democracy","Military Dict",1974,5,1
"211","Brazil",140,140,"South America","Americas","Joao Figueiredo","Joao Figueiredo.Brazil.1979.1984.Military Dict","Non-democracy","Military Dict",1979,6,1
"212","Brazil",140,140,"South America","Americas","Jose Sarney","Jose Sarney.Brazil.1985.1989.Presidential Dem","Democracy","Presidential Dem",1985,5,1
"213","Brazil",140,140,"South America","Americas","Fernando Collor de Mello","Fernando Collor de Mello.Brazil.1990.1991.Presidential Dem","Democracy","Presidential Dem",1990,2,1
"214","Brazil",140,140,"South America","Americas","Itamar Franco","Itamar Franco.Brazil.1992.1994.Presidential Dem","Democracy","Presidential Dem",1992,3,1
"215","Brazil",140,140,"South America","Americas","Fernando Henrique Cardoso","Fernando Henrique Cardoso.Brazil.1995.2002.Presidential Dem","Democracy","Presidential Dem",1995,8,1
"216","Brazil",140,140,"South America","Americas","Luiz Inï¿½cio Lula da Silva","Luiz Inï¿½cio Lula da Silva.Brazil.2003.2008.Presidential Dem","Democracy","Presidential Dem",2003,6,0
"217","Brunei Darussalam",835,835,"South-Eastern Asia","Asia","Hassanal Bolkiah","Hassanal Bolkiah.Brunei Darussalam.1984.2008.Monarchy","Non-democracy","Monarchy",1984,25,0
"218","Bulgaria",355,355,"Eastern Europe","Europe","Georgi Dimitrov","Georgi Dimitrov.Bulgaria.1946.1948.Civilian Dict","Non-democracy","Civilian Dict",1946,3,0
"219","Bulgaria",355,355,"Eastern Europe","Europe","Vasil Petrov Kolarov","Vasil Petrov Kolarov.Bulgaria.1949.1949.Civilian Dict","Non-democracy","Civilian Dict",1949,1,0
"220","Bulgaria",355,355,"Eastern Europe","Europe","Vulko Chervenkov","Vulko Chervenkov.Bulgaria.1950.1953.Civilian Dict","Non-democracy","Civilian Dict",1950,4,1
"221","Bulgaria",355,355,"Eastern Europe","Europe","Todor Zhivkov","Todor Zhivkov.Bulgaria.1954.1988.Civilian Dict","Non-democracy","Civilian Dict",1954,35,1
"222","Bulgaria",355,355,"Eastern Europe","Europe","Peter Mladenov","Peter Mladenov.Bulgaria.1989.1989.Civilian Dict","Non-democracy","Civilian Dict",1989,1,1
"223","Bulgaria",355,355,"Eastern Europe","Europe","Dimitur Popov","Dimitur Popov.Bulgaria.1990.1990.Mixed Dem","Democracy","Mixed Dem",1990,1,1
"224","Bulgaria",355,355,"Eastern Europe","Europe","Filip Dimitrov","Filip Dimitrov.Bulgaria.1991.1991.Mixed Dem","Democracy","Mixed Dem",1991,1,1
"225","Bulgaria",355,355,"Eastern Europe","Europe","Lyuben Berov","Lyuben Berov.Bulgaria.1992.1993.Mixed Dem","Democracy","Mixed Dem",1992,2,1
"226","Bulgaria",355,355,"Eastern Europe","Europe","Reneta Indzhova","Reneta Indzhova.Bulgaria.1994.1994.Mixed Dem","Democracy","Mixed Dem",1994,1,1
"227","Bulgaria",355,355,"Eastern Europe","Europe","Zhan Videnov","Zhan Videnov.Bulgaria.1995.1996.Mixed Dem","Democracy","Mixed Dem",1995,2,1
"228","Bulgaria",355,355,"Eastern Europe","Europe","Ivan Kostov","Ivan Kostov.Bulgaria.1997.2000.Mixed Dem","Democracy","Mixed Dem",1997,4,1
"229","Bulgaria",355,355,"Eastern Europe","Europe","Simeon Borisov Sakskoburggotski","Simeon Borisov Sakskoburggotski.Bulgaria.2001.2004.Mixed Dem","Democracy","Mixed Dem",2001,4,1
"230","Bulgaria",355,355,"Eastern Europe","Europe","Sergey Dimitrievich Stanishev","Sergey Dimitrievich Stanishev.Bulgaria.2005.2008.Mixed Dem","Democracy","Mixed Dem",2005,4,0
"231","Burkina Faso",439,439,"Western Africa","Africa","Maurice Yameogo","Maurice Yameogo.Burkina Faso.1960.1965.Civilian Dict","Non-democracy","Civilian Dict",1960,6,1
"232","Burkina Faso",439,439,"Western Africa","Africa","Sangoule Lamizana","Sangoule Lamizana.Burkina Faso.1966.1979.Military Dict","Non-democracy","Military Dict",1966,14,1
"233","Burkina Faso",439,439,"Western Africa","Africa","Saye Zerbo","Saye Zerbo.Burkina Faso.1980.1981.Military Dict","Non-democracy","Military Dict",1980,2,1
"234","Burkina Faso",439,439,"Western Africa","Africa","Jean-Baptiste Ouedraogo","Jean-Baptiste Ouedraogo.Burkina Faso.1982.1982.Military Dict","Non-democracy","Military Dict",1982,1,1
"235","Burkina Faso",439,439,"Western Africa","Africa","Thomas Sankara","Thomas Sankara.Burkina Faso.1983.1986.Military Dict","Non-democracy","Military Dict",1983,4,0
"236","Burkina Faso",439,439,"Western Africa","Africa","Blaise Compaor","Blaise Compaor.Burkina Faso.1987.2008.Military Dict","Non-democracy","Military Dict",1987,22,0
"237","Burundi",516,516,"Eastern Africa","Africa","Mwambutsa IV","Mwambutsa IV.Burundi.1962.1965.Monarchy","Non-democracy","Monarchy",1962,4,1
"238","Burundi",516,516,"Eastern Africa","Africa","Michel Micombero","Michel Micombero.Burundi.1966.1975.Military Dict","Non-democracy","Military Dict",1966,10,1
"239","Burundi",516,516,"Eastern Africa","Africa","Jean-Baptiste Bagaza","Jean-Baptiste Bagaza.Burundi.1976.1986.Military Dict","Non-democracy","Military Dict",1976,11,1
"240","Burundi",516,516,"Eastern Africa","Africa","Pierre Buyoya","Pierre Buyoya.Burundi.1987.1992.Military Dict","Non-democracy","Military Dict",1987,6,1
"241","Burundi",516,516,"Eastern Africa","Africa","Sylvie Kinigi","Sylvie Kinigi.Burundi.1993.1993.Presidential Dem","Democracy","Presidential Dem",1993,1,1
"242","Burundi",516,516,"Eastern Africa","Africa","Sylvestre Ntibantunganya","Sylvestre Ntibantunganya.Burundi.1994.1995.Presidential Dem","Democracy","Presidential Dem",1994,2,1
"243","Burundi",516,516,"Eastern Africa","Africa","Pierre Buyoya","Pierre Buyoya.Burundi.1996.2002.Military Dict","Non-democracy","Military Dict",1996,7,1
"244","Burundi",516,516,"Eastern Africa","Africa","Domitien Ndayizeye","Domitien Ndayizeye.Burundi.2003.2004.Civilian Dict","Non-democracy","Civilian Dict",2003,2,1
"245","Burundi",516,516,"Eastern Africa","Africa","Pierre Nkurunziza","Pierre Nkurunziza.Burundi.2005.2008.Presidential Dem","Democracy","Presidential Dem",2005,4,0
"246","Cambodia",811,811,"South-Eastern Asia","Asia","Norodom Sihanouk","Norodom Sihanouk.Cambodia.1953.1954.Monarchy","Non-democracy","Monarchy",1953,2,1
"247","Cambodia",811,811,"South-Eastern Asia","Asia","Norodom Sihanouk","Norodom Sihanouk.Cambodia.1955.1965.Civilian Dict","Non-democracy","Civilian Dict",1955,11,1
"248","Cambodia",811,811,"South-Eastern Asia","Asia","Lon Nol","Lon Nol.Cambodia.1966.1966.Military Dict","Non-democracy","Military Dict",1966,1,1
"249","Cambodia",811,811,"South-Eastern Asia","Asia","Norodom Sihanouk","Norodom Sihanouk.Cambodia.1967.1968.Civilian Dict","Non-democracy","Civilian Dict",1967,2,1
"250","Cambodia",811,811,"South-Eastern Asia","Asia","Lon Nol","Lon Nol.Cambodia.1969.1974.Military Dict","Non-democracy","Military Dict",1969,6,1
"251","Cambodia",811,811,"South-Eastern Asia","Asia","Norodom Sihanouk","Norodom Sihanouk.Cambodia.1975.1975.Civilian Dict","Non-democracy","Civilian Dict",1975,1,1
"252","Cambodia",811,811,"South-Eastern Asia","Asia","Pol Pot","Pol Pot.Cambodia.1976.1978.Civilian Dict","Non-democracy","Civilian Dict",1976,3,1
"253","Cambodia",811,811,"South-Eastern Asia","Asia","Heng Samrin","Heng Samrin.Cambodia.1979.1990.Military Dict","Non-democracy","Military Dict",1979,12,1
"254","Cambodia",811,811,"South-Eastern Asia","Asia","Norodom Sihanouk","Norodom Sihanouk.Cambodia.1991.1992.Civilian Dict","Non-democracy","Civilian Dict",1991,2,1
"255","Cambodia",811,811,"South-Eastern Asia","Asia","Hun Sen + Ranariddh","Hun Sen + Ranariddh.Cambodia.1993.1996.Civilian Dict","Non-democracy","Civilian Dict",1993,4,1
"256","Cambodia",811,811,"South-Eastern Asia","Asia","Hun Sen","Hun Sen.Cambodia.1997.2008.Civilian Dict","Non-democracy","Civilian Dict",1997,12,0
"257","Cameroon",471,471,"Middle Africa","Africa","Ahmadou Ahidjo","Ahmadou Ahidjo.Cameroon.1960.1981.Civilian Dict","Non-democracy","Civilian Dict",1960,22,1
"258","Cameroon",471,471,"Middle Africa","Africa","Paul Biya","Paul Biya.Cameroon.1982.2008.Civilian Dict","Non-democracy","Civilian Dict",1982,27,0
"259","Canada",20,20,"Northern America","Americas","W. MacKenzie King","W. MacKenzie King.Canada.1946.1947.Parliamentary Dem","Democracy","Parliamentary Dem",1946,2,1
"260","Canada",20,20,"Northern America","Americas","Louis Saint Laurent","Louis Saint Laurent.Canada.1948.1956.Parliamentary Dem","Democracy","Parliamentary Dem",1948,9,1
"261","Canada",20,20,"Northern America","Americas","John Diefenbaker","John Diefenbaker.Canada.1957.1962.Parliamentary Dem","Democracy","Parliamentary Dem",1957,6,1
"262","Canada",20,20,"Northern America","Americas","Lester Pearson","Lester Pearson.Canada.1963.1967.Parliamentary Dem","Democracy","Parliamentary Dem",1963,5,1
"263","Canada",20,20,"Northern America","Americas","Pierre Elliott Trudeau","Pierre Elliott Trudeau.Canada.1968.1978.Parliamentary Dem","Democracy","Parliamentary Dem",1968,11,1
"264","Canada",20,20,"Northern America","Americas","Joe Clark","Joe Clark.Canada.1979.1979.Parliamentary Dem","Democracy","Parliamentary Dem",1979,1,1
"265","Canada",20,20,"Northern America","Americas","Pierre Elliott Trudeau","Pierre Elliott Trudeau.Canada.1980.1983.Parliamentary Dem","Democracy","Parliamentary Dem",1980,4,1
"266","Canada",20,20,"Northern America","Americas","Brian Mulroney","Brian Mulroney.Canada.1984.1992.Parliamentary Dem","Democracy","Parliamentary Dem",1984,9,1
"267","Canada",20,20,"Northern America","Americas","Jean Chretien","Jean Chretien.Canada.1993.2002.Parliamentary Dem","Democracy","Parliamentary Dem",1993,10,1
"268","Canada",20,20,"Northern America","Americas","Paul Joseph Martin, Jr.","Paul Joseph Martin, Jr..Canada.2003.2005.Parliamentary Dem","Democracy","Parliamentary Dem",2003,3,1
"269","Canada",20,20,"Northern America","Americas","Stephen Joseph Harper","Stephen Joseph Harper.Canada.2006.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2006,3,0
"270","Cape Verde",402,402,"Western Africa","Africa","Aristides Pereira","Aristides Pereira.Cape Verde.1975.1989.Civilian Dict","Non-democracy","Civilian Dict",1975,15,1
"271","Cape Verde",402,402,"Western Africa","Africa","Aristides Pereira","Aristides Pereira.Cape Verde.1990.1990.Mixed Dem","Democracy","Mixed Dem",1990,1,1
"272","Cape Verde",402,402,"Western Africa","Africa","Carlos Veiga","Carlos Veiga.Cape Verde.1991.1999.Mixed Dem","Democracy","Mixed Dem",1991,9,1
"273","Cape Verde",402,402,"Western Africa","Africa","Antonio Gualberto do Rosario","Antonio Gualberto do Rosario.Cape Verde.2000.2000.Mixed Dem","Democracy","Mixed Dem",2000,1,1
"274","Cape Verde",402,402,"Western Africa","Africa","Jose Neves","Jose Neves.Cape Verde.2001.2008.Mixed Dem","Democracy","Mixed Dem",2001,8,0
"275","Central African Republic",482,482,"Middle Africa","Africa","David Dacko","David Dacko.Central African Republic.1960.1965.Civilian Dict","Non-democracy","Civilian Dict",1960,6,1
"276","Central African Republic",482,482,"Middle Africa","Africa","Jean-Bedel Bokassa","Jean-Bedel Bokassa.Central African Republic.1966.1978.Military Dict","Non-democracy","Military Dict",1966,13,1
"277","Central African Republic",482,482,"Middle Africa","Africa","David Dacko","David Dacko.Central African Republic.1979.1980.Civilian Dict","Non-democracy","Civilian Dict",1979,2,1
"278","Central African Republic",482,482,"Middle Africa","Africa","Andre Kolingba","Andre Kolingba.Central African Republic.1981.1992.Military Dict","Non-democracy","Military Dict",1981,12,1
"279","Central African Republic",482,482,"Middle Africa","Africa","Jean-Luc Mandaba","Jean-Luc Mandaba.Central African Republic.1993.1994.Mixed Dem","Democracy","Mixed Dem",1993,2,1
"280","Central African Republic",482,482,"Middle Africa","Africa","Gabriel Koyambounou","Gabriel Koyambounou.Central African Republic.1995.1995.Mixed Dem","Democracy","Mixed Dem",1995,1,1
"281","Central African Republic",482,482,"Middle Africa","Africa","Jean-Paul Ngoupande","Jean-Paul Ngoupande.Central African Republic.1996.1996.Mixed Dem","Democracy","Mixed Dem",1996,1,1
"282","Central African Republic",482,482,"Middle Africa","Africa","Michel Gbezera-Bria","Michel Gbezera-Bria.Central African Republic.1997.1998.Mixed Dem","Democracy","Mixed Dem",1997,2,1
"283","Central African Republic",482,482,"Middle Africa","Africa","Anicet Georges Dologuele","Anicet Georges Dologuele.Central African Republic.1999.2000.Mixed Dem","Democracy","Mixed Dem",1999,2,1
"284","Central African Republic",482,482,"Middle Africa","Africa","Martin Ziguele","Martin Ziguele.Central African Republic.2001.2002.Mixed Dem","Democracy","Mixed Dem",2001,2,1
"285","Central African Republic",482,482,"Middle Africa","Africa","Franï¿½ois Bozizï¿½","Franï¿½ois Bozizï¿½.Central African Republic.2003.2008.Military Dict","Non-democracy","Military Dict",2003,6,0
"286","Chad",483,483,"Middle Africa","Africa","Ngarta Tombolbaye","Ngarta Tombolbaye.Chad.1960.1974.Civilian Dict","Non-democracy","Civilian Dict",1960,15,0
"287","Chad",483,483,"Middle Africa","Africa","Felix Malloum","Felix Malloum.Chad.1975.1978.Military Dict","Non-democracy","Military Dict",1975,4,1
"288","Chad",483,483,"Middle Africa","Africa","Goukouni Ouedei","Goukouni Ouedei.Chad.1979.1981.Civilian Dict","Non-democracy","Civilian Dict",1979,3,1
"289","Chad",483,483,"Middle Africa","Africa","Hissene Habre","Hissene Habre.Chad.1982.1989.Civilian Dict","Non-democracy","Civilian Dict",1982,8,1
"290","Chad",483,483,"Middle Africa","Africa","Idriss Deby","Idriss Deby.Chad.1990.2008.Military Dict","Non-democracy","Military Dict",1990,19,0
"291","Chile",155,155,"South America","Americas","Gabriel Gonzalez Videla","Gabriel Gonzalez Videla.Chile.1946.1951.Presidential Dem","Democracy","Presidential Dem",1946,6,1
"292","Chile",155,155,"South America","Americas","Carlos Ibanez del Campo","Carlos Ibanez del Campo.Chile.1952.1957.Presidential Dem","Democracy","Presidential Dem",1952,6,1
"293","Chile",155,155,"South America","Americas","Jorge Alessandri Rodriguez","Jorge Alessandri Rodriguez.Chile.1958.1963.Presidential Dem","Democracy","Presidential Dem",1958,6,1
"294","Chile",155,155,"South America","Americas","Eduardo Frei Montalva","Eduardo Frei Montalva.Chile.1964.1969.Presidential Dem","Democracy","Presidential Dem",1964,6,1
"295","Chile",155,155,"South America","Americas","Salvador Allende Gossens","Salvador Allende Gossens.Chile.1970.1972.Presidential Dem","Democracy","Presidential Dem",1970,3,0
"296","Chile",155,155,"South America","Americas","Augusto Pinochet Ugarte","Augusto Pinochet Ugarte.Chile.1973.1989.Military Dict","Non-democracy","Military Dict",1973,17,1
"297","Chile",155,155,"South America","Americas","Patricio Aylwin Azacar","Patricio Aylwin Azacar.Chile.1990.1993.Presidential Dem","Democracy","Presidential Dem",1990,4,1
"298","Chile",155,155,"South America","Americas","Eduardo Frei Ruiz-Tagle","Eduardo Frei Ruiz-Tagle.Chile.1994.1999.Presidential Dem","Democracy","Presidential Dem",1994,6,1
"299","Chile",155,155,"South America","Americas","Ricardo Froilan Lagos Escobar","Ricardo Froilan Lagos Escobar.Chile.2000.2005.Presidential Dem","Democracy","Presidential Dem",2000,6,1
"300","Chile",155,155,"South America","Americas","Verï¿½nica Michelle Bachelet Jeria","Verï¿½nica Michelle Bachelet Jeria.Chile.2006.2008.Presidential Dem","Democracy","Presidential Dem",2006,3,0
"301","China",710,710,"Eastern Asia","Asia","Chiang Kai-Shek","Chiang Kai-Shek.China.1946.1948.Military Dict","Non-democracy","Military Dict",1946,3,1
"302","China",710,710,"Eastern Asia","Asia","Mao Zedong","Mao Zedong.China.1949.1975.Civilian Dict","Non-democracy","Civilian Dict",1949,27,0
"303","China",710,710,"Eastern Asia","Asia","Hua Guofeng","Hua Guofeng.China.1976.1977.Civilian Dict","Non-democracy","Civilian Dict",1976,2,1
"304","China",710,710,"Eastern Asia","Asia","Deng Xiaoping","Deng Xiaoping.China.1978.1996.Civilian Dict","Non-democracy","Civilian Dict",1978,19,0
"305","China",710,710,"Eastern Asia","Asia","Jiang Zemin","Jiang Zemin.China.1997.2001.Civilian Dict","Non-democracy","Civilian Dict",1997,5,1
"306","China",710,710,"Eastern Asia","Asia","Hu Jintao","Hu Jintao.China.2002.2008.Civilian Dict","Non-democracy","Civilian Dict",2002,7,0
"307","Colombia",100,100,"South America","Americas","Luis Mariano Ospina Perez","Luis Mariano Ospina Perez.Colombia.1946.1948.Presidential Dem","Democracy","Presidential Dem",1946,3,1
"308","Colombia",100,100,"South America","Americas","Luis Mariano Ospina Perez","Luis Mariano Ospina Perez.Colombia.1949.1949.Civilian Dict","Non-democracy","Civilian Dict",1949,1,1
"309","Colombia",100,100,"South America","Americas","Laureano Gomez Castro","Laureano Gomez Castro.Colombia.1950.1950.Civilian Dict","Non-democracy","Civilian Dict",1950,1,1
"310","Colombia",100,100,"South America","Americas","Roberto Urdaneta Arbelaez","Roberto Urdaneta Arbelaez.Colombia.1951.1952.Civilian Dict","Non-democracy","Civilian Dict",1951,2,1
"311","Colombia",100,100,"South America","Americas","Gustavo Rojas Pinilla","Gustavo Rojas Pinilla.Colombia.1953.1956.Military Dict","Non-democracy","Military Dict",1953,4,1
"312","Colombia",100,100,"South America","Americas","Gabriel Paris","Gabriel Paris.Colombia.1957.1957.Military Dict","Non-democracy","Military Dict",1957,1,1
"313","Colombia",100,100,"South America","Americas","Alberto Lleras Camargo","Alberto Lleras Camargo.Colombia.1958.1961.Presidential Dem","Democracy","Presidential Dem",1958,4,1
"314","Colombia",100,100,"South America","Americas","Guillermo Valencia","Guillermo Valencia.Colombia.1962.1965.Presidential Dem","Democracy","Presidential Dem",1962,4,1
"315","Colombia",100,100,"South America","Americas","Carlos Lleras Restrepo","Carlos Lleras Restrepo.Colombia.1966.1969.Presidential Dem","Democracy","Presidential Dem",1966,4,1
"316","Colombia",100,100,"South America","Americas","Misael Pastrana Borrero","Misael Pastrana Borrero.Colombia.1970.1973.Presidential Dem","Democracy","Presidential Dem",1970,4,1
"317","Colombia",100,100,"South America","Americas","Alfonso Lopez Michelson","Alfonso Lopez Michelson.Colombia.1974.1977.Presidential Dem","Democracy","Presidential Dem",1974,4,1
"318","Colombia",100,100,"South America","Americas","Julio Turbay Ayala","Julio Turbay Ayala.Colombia.1978.1981.Presidential Dem","Democracy","Presidential Dem",1978,4,1
"319","Colombia",100,100,"South America","Americas","Belisario Betancur Cuartas","Belisario Betancur Cuartas.Colombia.1982.1985.Presidential Dem","Democracy","Presidential Dem",1982,4,1
"320","Colombia",100,100,"South America","Americas","Virgilio Barco Vargas","Virgilio Barco Vargas.Colombia.1986.1989.Presidential Dem","Democracy","Presidential Dem",1986,4,1
"321","Colombia",100,100,"South America","Americas","Cesar Gaviria Trujillo","Cesar Gaviria Trujillo.Colombia.1990.1993.Presidential Dem","Democracy","Presidential Dem",1990,4,1
"322","Colombia",100,100,"South America","Americas","Ernesto Samper Pizano","Ernesto Samper Pizano.Colombia.1994.1997.Presidential Dem","Democracy","Presidential Dem",1994,4,1
"323","Colombia",100,100,"South America","Americas","Andres Pastrana Arango","Andres Pastrana Arango.Colombia.1998.2001.Presidential Dem","Democracy","Presidential Dem",1998,4,1
"324","Colombia",100,100,"South America","Americas","Alvaro Uribe Velez","Alvaro Uribe Velez.Colombia.2002.2008.Presidential Dem","Democracy","Presidential Dem",2002,7,0
"325","Comoros",581,581,"Eastern Africa","Africa","Ali Soilih","Ali Soilih.Comoros.1975.1977.Civilian Dict","Non-democracy","Civilian Dict",1975,3,0
"326","Comoros",581,581,"Eastern Africa","Africa","Ahmed Abdallah","Ahmed Abdallah.Comoros.1978.1988.Civilian Dict","Non-democracy","Civilian Dict",1978,11,0
"327","Comoros",581,581,"Eastern Africa","Africa","Said Mohammed Djohar","Said Mohammed Djohar.Comoros.1989.1989.Civilian Dict","Non-democracy","Civilian Dict",1989,1,1
"328","Comoros",581,581,"Eastern Africa","Africa","Said Mohammed Djohar","Said Mohammed Djohar.Comoros.1990.1992.Mixed Dem","Democracy","Mixed Dem",1990,3,1
"329","Comoros",581,581,"Eastern Africa","Africa","Ahmed Ben Cheikh Attoumane","Ahmed Ben Cheikh Attoumane.Comoros.1993.1993.Mixed Dem","Democracy","Mixed Dem",1993,1,1
"330","Comoros",581,581,"Eastern Africa","Africa","Halifa Houmadi","Halifa Houmadi.Comoros.1994.1994.Mixed Dem","Democracy","Mixed Dem",1994,1,1
"331","Comoros",581,581,"Eastern Africa","Africa","Caambi el-Yachourtu","Caambi el-Yachourtu.Comoros.1995.1995.Civilian Dict","Non-democracy","Civilian Dict",1995,1,1
"332","Comoros",581,581,"Eastern Africa","Africa","Mohammed Taki Abdoulkarim","Mohammed Taki Abdoulkarim.Comoros.1996.1997.Civilian Dict","Non-democracy","Civilian Dict",1996,2,0
"333","Comoros",581,581,"Eastern Africa","Africa","Tadjidine Ben Said Massounde","Tadjidine Ben Said Massounde.Comoros.1998.1998.Civilian Dict","Non-democracy","Civilian Dict",1998,1,1
"334","Comoros",581,581,"Eastern Africa","Africa","Azali Assoumani","Azali Assoumani.Comoros.1999.2003.Military Dict","Non-democracy","Military Dict",1999,5,1
"335","Comoros",581,581,"Eastern Africa","Africa","Azali Assoumani","Azali Assoumani.Comoros.2004.2005.Presidential Dem","Democracy","Presidential Dem",2004,2,1
"336","Comoros",581,581,"Eastern Africa","Africa","Ahmed Abdallah Sambi","Ahmed Abdallah Sambi.Comoros.2006.2008.Presidential Dem","Democracy","Presidential Dem",2006,3,0
"337","Congo (Brazzaville, Republic of Congo)",484,484,"Middle Africa","Africa","Fulbert Youlou","Fulbert Youlou.Congo (Brazzaville, Republic of Congo).1960.1962.Presidential Dem","Democracy","Presidential Dem",1960,3,1
"338","Congo (Brazzaville, Republic of Congo)",484,484,"Middle Africa","Africa","Alphonse Massamba-Debat","Alphonse Massamba-Debat.Congo (Brazzaville, Republic of Congo).1963.1967.Civilian Dict","Non-democracy","Civilian Dict",1963,5,1
"339","Congo (Brazzaville, Republic of Congo)",484,484,"Middle Africa","Africa","Alfred Raoul","Alfred Raoul.Congo (Brazzaville, Republic of Congo).1968.1968.Military Dict","Non-democracy","Military Dict",1968,1,1
"340","Congo (Brazzaville, Republic of Congo)",484,484,"Middle Africa","Africa","Marien Ngouabi","Marien Ngouabi.Congo (Brazzaville, Republic of Congo).1969.1976.Military Dict","Non-democracy","Military Dict",1969,8,0
"341","Congo (Brazzaville, Republic of Congo)",484,484,"Middle Africa","Africa","Joachim Yhombi-Opango","Joachim Yhombi-Opango.Congo (Brazzaville, Republic of Congo).1977.1978.Military Dict","Non-democracy","Military Dict",1977,2,1
"342","Congo (Brazzaville, Republic of Congo)",484,484,"Middle Africa","Africa","Denis Sassou-Nguesso","Denis Sassou-Nguesso.Congo (Brazzaville, Republic of Congo).1979.1991.Military Dict","Non-democracy","Military Dict",1979,13,1
"343","Congo (Brazzaville, Republic of Congo)",484,484,"Middle Africa","Africa","Claude-Antoine Dacosta","Claude-Antoine Dacosta.Congo (Brazzaville, Republic of Congo).1992.1992.Mixed Dem","Democracy","Mixed Dem",1992,1,1
"344","Congo (Brazzaville, Republic of Congo)",484,484,"Middle Africa","Africa","Jacques-Joachim Yhombi-Opango","Jacques-Joachim Yhombi-Opango.Congo (Brazzaville, Republic of Congo).1993.1995.Mixed Dem","Democracy","Mixed Dem",1993,3,1
"345","Congo (Brazzaville, Republic of Congo)",484,484,"Middle Africa","Africa","Charles David Ganao","Charles David Ganao.Congo (Brazzaville, Republic of Congo).1996.1996.Mixed Dem","Democracy","Mixed Dem",1996,1,1
"346","Congo (Brazzaville, Republic of Congo)",484,484,"Middle Africa","Africa","Denis Sassou-Nguesso","Denis Sassou-Nguesso.Congo (Brazzaville, Republic of Congo).1997.2008.Military Dict","Non-democracy","Military Dict",1997,12,0
"347","Costa Rica",94,94,"Central America","Americas","Teodoro Picado Michalski","Teodoro Picado Michalski.Costa Rica.1946.1947.Presidential Dem","Democracy","Presidential Dem",1946,2,1
"348","Costa Rica",94,94,"Central America","Americas","Jose Figueres Ferrer","Jose Figueres Ferrer.Costa Rica.1948.1948.Civilian Dict","Non-democracy","Civilian Dict",1948,1,1
"349","Costa Rica",94,94,"Central America","Americas","Otilia Ulate Blanco","Otilia Ulate Blanco.Costa Rica.1949.1952.Presidential Dem","Democracy","Presidential Dem",1949,4,1
"350","Costa Rica",94,94,"Central America","Americas","Jose Figueres Ferrer","Jose Figueres Ferrer.Costa Rica.1953.1957.Presidential Dem","Democracy","Presidential Dem",1953,5,1
"351","Costa Rica",94,94,"Central America","Americas","Mario Echandi Jimenez","Mario Echandi Jimenez.Costa Rica.1958.1961.Presidential Dem","Democracy","Presidential Dem",1958,4,1
"352","Costa Rica",94,94,"Central America","Americas","Francisco Jose Orlich Bolmarich","Francisco Jose Orlich Bolmarich.Costa Rica.1962.1965.Presidential Dem","Democracy","Presidential Dem",1962,4,1
"353","Costa Rica",94,94,"Central America","Americas","Jose Joaquim Trejos Fernandez","Jose Joaquim Trejos Fernandez.Costa Rica.1966.1969.Presidential Dem","Democracy","Presidential Dem",1966,4,1
"354","Costa Rica",94,94,"Central America","Americas","Jose Figueres Ferrer","Jose Figueres Ferrer.Costa Rica.1970.1973.Presidential Dem","Democracy","Presidential Dem",1970,4,1
"355","Costa Rica",94,94,"Central America","Americas","Daniel Oduber Quires","Daniel Oduber Quires.Costa Rica.1974.1977.Presidential Dem","Democracy","Presidential Dem",1974,4,1
"356","Costa Rica",94,94,"Central America","Americas","Rodrigo Carazo Odio","Rodrigo Carazo Odio.Costa Rica.1978.1981.Presidential Dem","Democracy","Presidential Dem",1978,4,1
"357","Costa Rica",94,94,"Central America","Americas","Luis Alberto Monge Alverez","Luis Alberto Monge Alverez.Costa Rica.1982.1985.Presidential Dem","Democracy","Presidential Dem",1982,4,1
"358","Costa Rica",94,94,"Central America","Americas","Oscar Arias Sanchez","Oscar Arias Sanchez.Costa Rica.1986.1989.Presidential Dem","Democracy","Presidential Dem",1986,4,1
"359","Costa Rica",94,94,"Central America","Americas","Rafael Angel Calderon Fournier","Rafael Angel Calderon Fournier.Costa Rica.1990.1993.Presidential Dem","Democracy","Presidential Dem",1990,4,1
"360","Costa Rica",94,94,"Central America","Americas","Jose Maria Figueres Olsen","Jose Maria Figueres Olsen.Costa Rica.1994.1997.Presidential Dem","Democracy","Presidential Dem",1994,4,1
"361","Costa Rica",94,94,"Central America","Americas","Miguel Angel Rodriguez Echeverria","Miguel Angel Rodriguez Echeverria.Costa Rica.1998.2001.Presidential Dem","Democracy","Presidential Dem",1998,4,1
"362","Costa Rica",94,94,"Central America","Americas","Abel Pacheco de la Espriella","Abel Pacheco de la Espriella.Costa Rica.2002.2005.Presidential Dem","Democracy","Presidential Dem",2002,4,1
"363","Costa Rica",94,94,"Central America","Americas","ï¿½scar Rafael Arias Sï¿½nchez","ï¿½scar Rafael Arias Sï¿½nchez.Costa Rica.2006.2008.Presidential Dem","Democracy","Presidential Dem",2006,3,0
"364","Cote d'Ivoire",437,437,"Western Africa","Africa","Felix Houphouet-Boigny","Felix Houphouet-Boigny.Cote d'Ivoire.1960.1992.Civilian Dict","Non-democracy","Civilian Dict",1960,33,0
"365","Cote d'Ivoire",437,437,"Western Africa","Africa","Henri Konan Bedie","Henri Konan Bedie.Cote d'Ivoire.1993.1998.Civilian Dict","Non-democracy","Civilian Dict",1993,6,1
"366","Cote d'Ivoire",437,437,"Western Africa","Africa","Robert Guei","Robert Guei.Cote d'Ivoire.1999.1999.Military Dict","Non-democracy","Military Dict",1999,1,1
"367","Cote d'Ivoire",437,437,"Western Africa","Africa","Laurent Gbagbo","Laurent Gbagbo.Cote d'Ivoire.2000.2008.Civilian Dict","Non-democracy","Civilian Dict",2000,9,0
"368","Croatia",344,344,"Southern Europe","Europe","Franjo Greguric","Franjo Greguric.Croatia.1991.1991.Mixed Dem","Democracy","Mixed Dem",1991,1,1
"369","Croatia",344,344,"Southern Europe","Europe","Hrvoje Sarinic","Hrvoje Sarinic.Croatia.1992.1992.Mixed Dem","Democracy","Mixed Dem",1992,1,1
"370","Croatia",344,344,"Southern Europe","Europe","Nikica Valentic","Nikica Valentic.Croatia.1993.1994.Mixed Dem","Democracy","Mixed Dem",1993,2,1
"371","Croatia",344,344,"Southern Europe","Europe","Zlatko Matesa","Zlatko Matesa.Croatia.1995.1999.Mixed Dem","Democracy","Mixed Dem",1995,5,1
"372","Croatia",344,344,"Southern Europe","Europe","Ivica Racan","Ivica Racan.Croatia.2000.2002.Mixed Dem","Democracy","Mixed Dem",2000,3,1
"373","Croatia",344,344,"Southern Europe","Europe","Ivo Sanader","Ivo Sanader.Croatia.2003.2008.Mixed Dem","Democracy","Mixed Dem",2003,6,0
"374","Cuba",40,40,"Caribbean","Americas","Fulgencio Batista y Zaldivar","Fulgencio Batista y Zaldivar.Cuba.1946.1951.Presidential Dem","Democracy","Presidential Dem",1946,6,1
"375","Cuba",40,40,"Caribbean","Americas","Fulgencio Batista y Zaldivar","Fulgencio Batista y Zaldivar.Cuba.1952.1958.Military Dict","Non-democracy","Military Dict",1952,7,1
"376","Cuba",40,40,"Caribbean","Americas","Fidel Castro Ruz","Fidel Castro Ruz.Cuba.1959.2005.Civilian Dict","Non-democracy","Civilian Dict",1959,47,1
"377","Cuba",40,40,"Caribbean","Americas","Raul Castro","Raul Castro.Cuba.2006.2008.Military Dict","Non-democracy","Military Dict",2006,3,0
"378","Cyprus",352,352,"Western Asia","Asia","Archbishop Makarios","Archbishop Makarios.Cyprus.1960.1976.Civilian Dict","Non-democracy","Civilian Dict",1960,17,0
"379","Cyprus",352,352,"Western Asia","Asia","Spyros Kyprianou","Spyros Kyprianou.Cyprus.1977.1982.Civilian Dict","Non-democracy","Civilian Dict",1977,6,0
"380","Cyprus",352,352,"Western Asia","Asia","Spyros Kyprianou","Spyros Kyprianou.Cyprus.1983.1987.Presidential Dem","Democracy","Presidential Dem",1983,5,1
"381","Cyprus",352,352,"Western Asia","Asia","George Vassiliou","George Vassiliou.Cyprus.1988.1992.Presidential Dem","Democracy","Presidential Dem",1988,5,1
"382","Cyprus",352,352,"Western Asia","Asia","Glafkos Klerides","Glafkos Klerides.Cyprus.1993.2002.Presidential Dem","Democracy","Presidential Dem",1993,10,1
"383","Cyprus",352,352,"Western Asia","Asia","Tassos Nikolaou Papadopoulos","Tassos Nikolaou Papadopoulos.Cyprus.2003.2007.Presidential Dem","Democracy","Presidential Dem",2003,5,1
"384","Cyprus",352,352,"Western Asia","Asia","Dimitris Christofi Christofias","Dimitris Christofi Christofias.Cyprus.2008.2008.Presidential Dem","Democracy","Presidential Dem",2008,1,0
"385","Czech Republic",316,316,"Eastern Europe","Europe","Vaclav Klaus","Vaclav Klaus.Czech Republic.1993.1996.Parliamentary Dem","Democracy","Parliamentary Dem",1993,4,1
"386","Czech Republic",316,316,"Eastern Europe","Europe","Josef Tosovsky","Josef Tosovsky.Czech Republic.1997.1997.Parliamentary Dem","Democracy","Parliamentary Dem",1997,1,1
"387","Czech Republic",316,316,"Eastern Europe","Europe","Milos Zeman","Milos Zeman.Czech Republic.1998.2001.Parliamentary Dem","Democracy","Parliamentary Dem",1998,4,1
"388","Czech Republic",316,316,"Eastern Europe","Europe","Vladimir Spidla","Vladimir Spidla.Czech Republic.2002.2003.Parliamentary Dem","Democracy","Parliamentary Dem",2002,2,1
"389","Czech Republic",316,316,"Eastern Europe","Europe","Stanislav Gross","Stanislav Gross.Czech Republic.2004.2004.Parliamentary Dem","Democracy","Parliamentary Dem",2004,1,1
"390","Czech Republic",316,316,"Eastern Europe","Europe","Jirï¿½ Paroubek","Jirï¿½ Paroubek.Czech Republic.2005.2005.Parliamentary Dem","Democracy","Parliamentary Dem",2005,1,1
"391","Czech Republic",316,316,"Eastern Europe","Europe","Mirek Topolï¿½nek","Mirek Topolï¿½nek.Czech Republic.2006.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2006,3,0
"392","Czechoslovakia",315,315,"Eastern Europe","Europe","Klement Gottwald","Klement Gottwald.Czechoslovakia.1946.1952.Civilian Dict","Non-democracy","Civilian Dict",1946,7,0
"393","Czechoslovakia",315,315,"Eastern Europe","Europe","Antonin Novotny","Antonin Novotny.Czechoslovakia.1953.1967.Civilian Dict","Non-democracy","Civilian Dict",1953,15,1
"394","Czechoslovakia",315,315,"Eastern Europe","Europe","Alexander Dubcek","Alexander Dubcek.Czechoslovakia.1968.1968.Civilian Dict","Non-democracy","Civilian Dict",1968,1,1
"395","Czechoslovakia",315,315,"Eastern Europe","Europe","Gustav Husak","Gustav Husak.Czechoslovakia.1969.1986.Civilian Dict","Non-democracy","Civilian Dict",1969,18,1
"396","Czechoslovakia",315,315,"Eastern Europe","Europe","Milos Jakes","Milos Jakes.Czechoslovakia.1987.1988.Civilian Dict","Non-democracy","Civilian Dict",1987,2,1
"397","Czechoslovakia",315,315,"Eastern Europe","Europe","Marian Calfa","Marian Calfa.Czechoslovakia.1989.1991.Parliamentary Dem","Democracy","Parliamentary Dem",1989,3,1
"398","Czechoslovakia",315,315,"Eastern Europe","Europe","Jan Strasky","Jan Strasky.Czechoslovakia.1992.1992.Parliamentary Dem","Democracy","Parliamentary Dem",1992,1,0
"399","Democratic Republic of the Congo (Zaire, Congo-Kinshasha)",490,490,"Middle Africa","Africa","Joseph Kasabuvu","Joseph Kasabuvu.Democratic Republic of the Congo (Zaire, Congo-Kinshasha).1960.1964.Civilian Dict","Non-democracy","Civilian Dict",1960,5,1
"400","Democratic Republic of the Congo (Zaire, Congo-Kinshasha)",490,490,"Middle Africa","Africa","Mobutu Sese Seko","Mobutu Sese Seko.Democratic Republic of the Congo (Zaire, Congo-Kinshasha).1965.1996.Military Dict","Non-democracy","Military Dict",1965,32,1
"401","Democratic Republic of the Congo (Zaire, Congo-Kinshasha)",490,490,"Middle Africa","Africa","Laurent Kabila","Laurent Kabila.Democratic Republic of the Congo (Zaire, Congo-Kinshasha).1997.2000.Civilian Dict","Non-democracy","Civilian Dict",1997,4,0
"402","Democratic Republic of the Congo (Zaire, Congo-Kinshasha)",490,490,"Middle Africa","Africa","Joseph Kabila","Joseph Kabila.Democratic Republic of the Congo (Zaire, Congo-Kinshasha).2001.2008.Civilian Dict","Non-democracy","Civilian Dict",2001,8,0
"403","Denmark",390,390,"Northern Europe","Europe","Knud Kristensen","Knud Kristensen.Denmark.1946.1946.Parliamentary Dem","Democracy","Parliamentary Dem",1946,1,1
"404","Denmark",390,390,"Northern Europe","Europe","Hans Hedtoft","Hans Hedtoft.Denmark.1947.1949.Parliamentary Dem","Democracy","Parliamentary Dem",1947,3,1
"405","Denmark",390,390,"Northern Europe","Europe","Erik Eriksen","Erik Eriksen.Denmark.1950.1952.Parliamentary Dem","Democracy","Parliamentary Dem",1950,3,1
"406","Denmark",390,390,"Northern Europe","Europe","Hans Hedtoft","Hans Hedtoft.Denmark.1953.1954.Parliamentary Dem","Democracy","Parliamentary Dem",1953,2,0
"407","Denmark",390,390,"Northern Europe","Europe","Hans Hansen","Hans Hansen.Denmark.1955.1959.Parliamentary Dem","Democracy","Parliamentary Dem",1955,5,0
"408","Denmark",390,390,"Northern Europe","Europe","Viggo Kampmann","Viggo Kampmann.Denmark.1960.1961.Parliamentary Dem","Democracy","Parliamentary Dem",1960,2,1
"409","Denmark",390,390,"Northern Europe","Europe","Jens-Otto Krag","Jens-Otto Krag.Denmark.1962.1967.Parliamentary Dem","Democracy","Parliamentary Dem",1962,6,1
"410","Denmark",390,390,"Northern Europe","Europe","Hilmar Baunsgaard","Hilmar Baunsgaard.Denmark.1968.1970.Parliamentary Dem","Democracy","Parliamentary Dem",1968,3,1
"411","Denmark",390,390,"Northern Europe","Europe","Jens-Otto Krag","Jens-Otto Krag.Denmark.1971.1971.Parliamentary Dem","Democracy","Parliamentary Dem",1971,1,1
"412","Denmark",390,390,"Northern Europe","Europe","Anker Jorgensen","Anker Jorgensen.Denmark.1972.1972.Parliamentary Dem","Democracy","Parliamentary Dem",1972,1,1
"413","Denmark",390,390,"Northern Europe","Europe","Poul Hartling","Poul Hartling.Denmark.1973.1974.Parliamentary Dem","Democracy","Parliamentary Dem",1973,2,1
"414","Denmark",390,390,"Northern Europe","Europe","Anker Jorgensen","Anker Jorgensen.Denmark.1975.1981.Parliamentary Dem","Democracy","Parliamentary Dem",1975,7,1
"415","Denmark",390,390,"Northern Europe","Europe","Poul Schluter","Poul Schluter.Denmark.1982.1992.Parliamentary Dem","Democracy","Parliamentary Dem",1982,11,1
"416","Denmark",390,390,"Northern Europe","Europe","Poul Nyrup Rasmussen","Poul Nyrup Rasmussen.Denmark.1993.2000.Parliamentary Dem","Democracy","Parliamentary Dem",1993,8,1
"417","Denmark",390,390,"Northern Europe","Europe","Anders Fogh Rasmussen","Anders Fogh Rasmussen.Denmark.2001.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2001,8,0
"418","Djibouti",522,522,"Eastern Africa","Africa","Hassan Gouled Aptidon","Hassan Gouled Aptidon.Djibouti.1977.1998.Civilian Dict","Non-democracy","Civilian Dict",1977,22,1
"419","Djibouti",522,522,"Eastern Africa","Africa","Ismail Omar Guelleh","Ismail Omar Guelleh.Djibouti.1999.2008.Civilian Dict","Non-democracy","Civilian Dict",1999,10,0
"420","Dominica",54,54,"Caribbean","Americas","Patrick John","Patrick John.Dominica.1978.1978.Parliamentary Dem","Democracy","Parliamentary Dem",1978,1,1
"421","Dominica",54,54,"Caribbean","Americas","Oliver Seraphine","Oliver Seraphine.Dominica.1979.1979.Parliamentary Dem","Democracy","Parliamentary Dem",1979,1,1
"422","Dominica",54,54,"Caribbean","Americas","Eugenia Charles","Eugenia Charles.Dominica.1980.1994.Parliamentary Dem","Democracy","Parliamentary Dem",1980,15,1
"423","Dominica",54,54,"Caribbean","Americas","Edison James","Edison James.Dominica.1995.1999.Parliamentary Dem","Democracy","Parliamentary Dem",1995,5,1
"424","Dominica",54,54,"Caribbean","Americas","Pierre Charles","Pierre Charles.Dominica.2000.2003.Parliamentary Dem","Democracy","Parliamentary Dem",2000,4,1
"425","Dominica",54,54,"Caribbean","Americas","Roosevelt Skerrit","Roosevelt Skerrit.Dominica.2004.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2004,5,0
"426","Dominican Republic",42,42,"Caribbean","Americas","Rafael Trujillo","Rafael Trujillo.Dominican Republic.1946.1960.Military Dict","Non-democracy","Military Dict",1946,15,0
"427","Dominican Republic",42,42,"Caribbean","Americas","Joaquin Balaguer","Joaquin Balaguer.Dominican Republic.1961.1961.Civilian Dict","Non-democracy","Civilian Dict",1961,1,1
"428","Dominican Republic",42,42,"Caribbean","Americas","Rafael Bonnelly","Rafael Bonnelly.Dominican Republic.1962.1962.Civilian Dict","Non-democracy","Civilian Dict",1962,1,1
"429","Dominican Republic",42,42,"Caribbean","Americas","military","military.Dominican Republic.1963.1964.Military Dict","Non-democracy","Military Dict",1963,2,1
"430","Dominican Republic",42,42,"Caribbean","Americas","Hector Garcia-Godoy Caceres","Hector Garcia-Godoy Caceres.Dominican Republic.1965.1965.Civilian Dict","Non-democracy","Civilian Dict",1965,1,1
"431","Dominican Republic",42,42,"Caribbean","Americas","Joaquin Balaguer","Joaquin Balaguer.Dominican Republic.1966.1977.Presidential Dem","Democracy","Presidential Dem",1966,12,1
"432","Dominican Republic",42,42,"Caribbean","Americas","Antonio Guzman Fernandez","Antonio Guzman Fernandez.Dominican Republic.1978.1981.Presidential Dem","Democracy","Presidential Dem",1978,4,0
"433","Dominican Republic",42,42,"Caribbean","Americas","Salvador Jorge Blanco","Salvador Jorge Blanco.Dominican Republic.1982.1985.Presidential Dem","Democracy","Presidential Dem",1982,4,1
"434","Dominican Republic",42,42,"Caribbean","Americas","Joaquin Balaguer","Joaquin Balaguer.Dominican Republic.1986.1995.Presidential Dem","Democracy","Presidential Dem",1986,10,1
"435","Dominican Republic",42,42,"Caribbean","Americas","Leonel Fernandez Reyna","Leonel Fernandez Reyna.Dominican Republic.1996.1999.Presidential Dem","Democracy","Presidential Dem",1996,4,1
"436","Dominican Republic",42,42,"Caribbean","Americas","Rafael Hipolito Mejia Dominguez","Rafael Hipolito Mejia Dominguez.Dominican Republic.2000.2003.Presidential Dem","Democracy","Presidential Dem",2000,4,1
"437","Dominican Republic",42,42,"Caribbean","Americas","Leonel Antonio Fernï¿½ndez Reyna","Leonel Antonio Fernï¿½ndez Reyna.Dominican Republic.2004.2008.Presidential Dem","Democracy","Presidential Dem",2004,5,0
"438","East Timor",860,860,"South-Eastern Asia","Asia","Marï¿½ Bim Amude Alkatiri","Marï¿½ Bim Amude Alkatiri.East Timor.2002.2005.Mixed Dem","Democracy","Mixed Dem",2002,4,1
"439","East Timor",860,860,"South-Eastern Asia","Asia","Josï¿½ Ramos-Horta","Josï¿½ Ramos-Horta.East Timor.2006.2006.Mixed Dem","Democracy","Mixed Dem",2006,1,1
"440","East Timor",860,860,"South-Eastern Asia","Asia","Kay Rala Xanana Gusmï¿½o","Kay Rala Xanana Gusmï¿½o.East Timor.2007.2008.Mixed Dem","Democracy","Mixed Dem",2007,2,0
"441","Ecuador",130,130,"South America","Americas","Jose Velasco Ibarra","Jose Velasco Ibarra.Ecuador.1946.1946.Presidential Dem","Democracy","Presidential Dem",1946,1,1
"442","Ecuador",130,130,"South America","Americas","military","military.Ecuador.1947.1947.Military Dict","Non-democracy","Military Dict",1947,1,1
"443","Ecuador",130,130,"South America","Americas","Galo Plaza Lasso","Galo Plaza Lasso.Ecuador.1948.1951.Presidential Dem","Democracy","Presidential Dem",1948,4,1
"444","Ecuador",130,130,"South America","Americas","Jose Velasco Ibarra","Jose Velasco Ibarra.Ecuador.1952.1955.Presidential Dem","Democracy","Presidential Dem",1952,4,1
"445","Ecuador",130,130,"South America","Americas","Camilo Ponce Enriquez","Camilo Ponce Enriquez.Ecuador.1956.1959.Presidential Dem","Democracy","Presidential Dem",1956,4,1
"446","Ecuador",130,130,"South America","Americas","Jose Velasco Ibarra","Jose Velasco Ibarra.Ecuador.1960.1960.Presidential Dem","Democracy","Presidential Dem",1960,1,1
"447","Ecuador",130,130,"South America","Americas","Carlos Arosemena Monroy","Carlos Arosemena Monroy.Ecuador.1961.1962.Presidential Dem","Democracy","Presidential Dem",1961,2,1
"448","Ecuador",130,130,"South America","Americas","Ramon Castro Jijon","Ramon Castro Jijon.Ecuador.1963.1965.Military Dict","Non-democracy","Military Dict",1963,3,1
"449","Ecuador",130,130,"South America","Americas","Otto Arosemena Gomez","Otto Arosemena Gomez.Ecuador.1966.1967.Civilian Dict","Non-democracy","Civilian Dict",1966,2,1
"450","Ecuador",130,130,"South America","Americas","Jose Velasco Ibarra","Jose Velasco Ibarra.Ecuador.1968.1971.Civilian Dict","Non-democracy","Civilian Dict",1968,4,1
"451","Ecuador",130,130,"South America","Americas","Guillermo Rodriguez Lara","Guillermo Rodriguez Lara.Ecuador.1972.1975.Military Dict","Non-democracy","Military Dict",1972,4,1
"452","Ecuador",130,130,"South America","Americas","Alfredo Poveda Burbano","Alfredo Poveda Burbano.Ecuador.1976.1978.Military Dict","Non-democracy","Military Dict",1976,3,1
"453","Ecuador",130,130,"South America","Americas","Jaime Roldos Aguillera","Jaime Roldos Aguillera.Ecuador.1979.1980.Presidential Dem","Democracy","Presidential Dem",1979,2,0
"454","Ecuador",130,130,"South America","Americas","Osvaldo Hurtado Larrea","Osvaldo Hurtado Larrea.Ecuador.1981.1983.Presidential Dem","Democracy","Presidential Dem",1981,3,1
"455","Ecuador",130,130,"South America","Americas","Leon Febres Cordero","Leon Febres Cordero.Ecuador.1984.1987.Presidential Dem","Democracy","Presidential Dem",1984,4,1
"456","Ecuador",130,130,"South America","Americas","Rodrigo Borja Cevallos","Rodrigo Borja Cevallos.Ecuador.1988.1991.Presidential Dem","Democracy","Presidential Dem",1988,4,1
"457","Ecuador",130,130,"South America","Americas","Sixto Duran Ballen","Sixto Duran Ballen.Ecuador.1992.1995.Presidential Dem","Democracy","Presidential Dem",1992,4,1
"458","Ecuador",130,130,"South America","Americas","Abdala Bucaram Ortiz","Abdala Bucaram Ortiz.Ecuador.1996.1996.Presidential Dem","Democracy","Presidential Dem",1996,1,1
"459","Ecuador",130,130,"South America","Americas","Fabian Alarcon","Fabian Alarcon.Ecuador.1997.1997.Presidential Dem","Democracy","Presidential Dem",1997,1,1
"460","Ecuador",130,130,"South America","Americas","Jamil Mahuad","Jamil Mahuad.Ecuador.1998.1999.Presidential Dem","Democracy","Presidential Dem",1998,2,1
"461","Ecuador",130,130,"South America","Americas","Gustavo Noboa Bejarano","Gustavo Noboa Bejarano.Ecuador.2000.2001.Civilian Dict","Non-democracy","Civilian Dict",2000,2,1
"462","Ecuador",130,130,"South America","Americas","Gustavo Noboa Bejarano","Gustavo Noboa Bejarano.Ecuador.2002.2002.Presidential Dem","Democracy","Presidential Dem",2002,1,1
"463","Ecuador",130,130,"South America","Americas","Lucio Gutiï¿½rrez","Lucio Gutiï¿½rrez.Ecuador.2003.2004.Presidential Dem","Democracy","Presidential Dem",2003,2,1
"464","Ecuador",130,130,"South America","Americas","Alfredo Palacio","Alfredo Palacio.Ecuador.2005.2006.Presidential Dem","Democracy","Presidential Dem",2005,2,1
"465","Ecuador",130,130,"South America","Americas","Rafael Correa","Rafael Correa.Ecuador.2007.2008.Presidential Dem","Democracy","Presidential Dem",2007,2,0
"466","Egypt",651,651,"Northern Africa","Africa","Faruq I","Faruq I.Egypt.1946.1951.Monarchy","Non-democracy","Monarchy",1946,6,1
"467","Egypt",651,651,"Northern Africa","Africa","Mohammed Naguib","Mohammed Naguib.Egypt.1952.1953.Military Dict","Non-democracy","Military Dict",1952,2,1
"468","Egypt",651,651,"Northern Africa","Africa","Gamal Nasser","Gamal Nasser.Egypt.1954.1969.Military Dict","Non-democracy","Military Dict",1954,16,0
"469","Egypt",651,651,"Northern Africa","Africa","Anwar el-Sadat","Anwar el-Sadat.Egypt.1970.1980.Military Dict","Non-democracy","Military Dict",1970,11,0
"470","Egypt",651,651,"Northern Africa","Africa","Hosni Mubarak","Hosni Mubarak.Egypt.1981.2008.Military Dict","Non-democracy","Military Dict",1981,28,0
"471","El Salvador",92,92,"Central America","Americas","Salvador Castaneda Castro","Salvador Castaneda Castro.El Salvador.1946.1947.Military Dict","Non-democracy","Military Dict",1946,2,1
"472","El Salvador",92,92,"Central America","Americas","Manuel Cordoba","Manuel Cordoba.El Salvador.1948.1948.Military Dict","Non-democracy","Military Dict",1948,1,1
"473","El Salvador",92,92,"Central America","Americas","Oscar Bolanos","Oscar Bolanos.El Salvador.1949.1949.Military Dict","Non-democracy","Military Dict",1949,1,1
"474","El Salvador",92,92,"Central America","Americas","Oscar Osorio","Oscar Osorio.El Salvador.1950.1955.Military Dict","Non-democracy","Military Dict",1950,6,1
"475","El Salvador",92,92,"Central America","Americas","Jose Lemus","Jose Lemus.El Salvador.1956.1959.Military Dict","Non-democracy","Military Dict",1956,4,1
"476","El Salvador",92,92,"Central America","Americas","Miguel Castillo","Miguel Castillo.El Salvador.1960.1960.Military Dict","Non-democracy","Military Dict",1960,1,1
"477","El Salvador",92,92,"Central America","Americas","Anibal Portillo","Anibal Portillo.El Salvador.1961.1961.Military Dict","Non-democracy","Military Dict",1961,1,1
"478","El Salvador",92,92,"Central America","Americas","Julio Rivera Carballo","Julio Rivera Carballo.El Salvador.1962.1966.Military Dict","Non-democracy","Military Dict",1962,5,1
"479","El Salvador",92,92,"Central America","Americas","Fidel Sanchez Hernandez","Fidel Sanchez Hernandez.El Salvador.1967.1971.Military Dict","Non-democracy","Military Dict",1967,5,1
"480","El Salvador",92,92,"Central America","Americas","Arturo Molina Barraza","Arturo Molina Barraza.El Salvador.1972.1976.Military Dict","Non-democracy","Military Dict",1972,5,1
"481","El Salvador",92,92,"Central America","Americas","Carlos Romero Mena","Carlos Romero Mena.El Salvador.1977.1978.Military Dict","Non-democracy","Military Dict",1977,2,1
"482","El Salvador",92,92,"Central America","Americas","Adolfo Majano + Jaime Gutierrez","Adolfo Majano + Jaime Gutierrez.El Salvador.1979.1979.Military Dict","Non-democracy","Military Dict",1979,1,1
"483","El Salvador",92,92,"Central America","Americas","Jose Napoleon Duarte","Jose Napoleon Duarte.El Salvador.1980.1981.Civilian Dict","Non-democracy","Civilian Dict",1980,2,1
"484","El Salvador",92,92,"Central America","Americas","military","military.El Salvador.1982.1983.Military Dict","Non-democracy","Military Dict",1982,2,1
"485","El Salvador",92,92,"Central America","Americas","Jose Napoleon Duarte","Jose Napoleon Duarte.El Salvador.1984.1988.Presidential Dem","Democracy","Presidential Dem",1984,5,1
"486","El Salvador",92,92,"Central America","Americas","Alfredo Christiani","Alfredo Christiani.El Salvador.1989.1993.Presidential Dem","Democracy","Presidential Dem",1989,5,1
"487","El Salvador",92,92,"Central America","Americas","Armando Calderon Sol","Armando Calderon Sol.El Salvador.1994.1998.Presidential Dem","Democracy","Presidential Dem",1994,5,1
"488","El Salvador",92,92,"Central America","Americas","Francisco Guillermo Flores Perez","Francisco Guillermo Flores Perez.El Salvador.1999.2003.Presidential Dem","Democracy","Presidential Dem",1999,5,1
"489","El Salvador",92,92,"Central America","Americas","Elï¿½as Antonio Saca Gonzï¿½lez","Elï¿½as Antonio Saca Gonzï¿½lez.El Salvador.2004.2008.Presidential Dem","Democracy","Presidential Dem",2004,5,0
"490","Equatorial Guinea",411,411,"Middle Africa","Africa","Francisco Macias Nguema","Francisco Macias Nguema.Equatorial Guinea.1968.1978.Civilian Dict","Non-democracy","Civilian Dict",1968,11,1
"491","Equatorial Guinea",411,411,"Middle Africa","Africa","Teodoro Obiang Nguema Mbasogo","Teodoro Obiang Nguema Mbasogo.Equatorial Guinea.1979.2008.Military Dict","Non-democracy","Military Dict",1979,30,0
"492","Eritrea",531,531,"Eastern Africa","Africa","Isaias Afwerki","Isaias Afwerki.Eritrea.1993.2008.Civilian Dict","Non-democracy","Civilian Dict",1993,16,0
"493","Estonia",366,366,"Northern Europe","Europe","Edgar Savisaar","Edgar Savisaar.Estonia.1991.1991.Parliamentary Dem","Democracy","Parliamentary Dem",1991,1,1
"494","Estonia",366,366,"Northern Europe","Europe","Mart Laar","Mart Laar.Estonia.1992.1993.Parliamentary Dem","Democracy","Parliamentary Dem",1992,2,1
"495","Estonia",366,366,"Northern Europe","Europe","Andres Tarand","Andres Tarand.Estonia.1994.1994.Parliamentary Dem","Democracy","Parliamentary Dem",1994,1,1
"496","Estonia",366,366,"Northern Europe","Europe","Tiit Vahi","Tiit Vahi.Estonia.1995.1996.Parliamentary Dem","Democracy","Parliamentary Dem",1995,2,1
"497","Estonia",366,366,"Northern Europe","Europe","Mart Siimann","Mart Siimann.Estonia.1997.1998.Parliamentary Dem","Democracy","Parliamentary Dem",1997,2,1
"498","Estonia",366,366,"Northern Europe","Europe","Mart Laar","Mart Laar.Estonia.1999.2001.Parliamentary Dem","Democracy","Parliamentary Dem",1999,3,1
"499","Estonia",366,366,"Northern Europe","Europe","Siim Kallas","Siim Kallas.Estonia.2002.2002.Parliamentary Dem","Democracy","Parliamentary Dem",2002,1,1
"500","Estonia",366,366,"Northern Europe","Europe","Juhan Parts","Juhan Parts.Estonia.2003.2004.Parliamentary Dem","Democracy","Parliamentary Dem",2003,2,1
"501","Estonia",366,366,"Northern Europe","Europe","Andrus Ansip","Andrus Ansip.Estonia.2005.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2005,4,0
"502","Ethiopia",530,530,"Eastern Africa","Africa","Haile Selassie","Haile Selassie.Ethiopia.1946.1973.Monarchy","Non-democracy","Monarchy",1946,28,1
"503","Ethiopia",530,530,"Eastern Africa","Africa","Teferi Benti","Teferi Benti.Ethiopia.1974.1976.Military Dict","Non-democracy","Military Dict",1974,3,1
"504","Ethiopia",530,530,"Eastern Africa","Africa","Mengistu Haile Mariam","Mengistu Haile Mariam.Ethiopia.1977.1990.Military Dict","Non-democracy","Military Dict",1977,14,1
"505","Ethiopia",530,530,"Eastern Africa","Africa","Meles Zenawi","Meles Zenawi.Ethiopia.1991.1994.Civilian Dict","Non-democracy","Civilian Dict",1991,4,0
"506","Ethiopia",530,529,"Eastern Africa","Africa","Meles Zenawi","Meles Zenawi.Ethiopia.1991.1994.Civilian Dict","Non-democracy","Civilian Dict",1991,4,0
"507","Ethiopia",530,529,"Eastern Africa","Africa","Negasso Gidada","Negasso Gidada.Ethiopia.1995.2000.Civilian Dict","Non-democracy","Civilian Dict",1995,6,1
"508","Ethiopia",530,529,"Eastern Africa","Africa","Girma Wolde-Giyorgis Lucha","Girma Wolde-Giyorgis Lucha.Ethiopia.2001.2008.Civilian Dict","Non-democracy","Civilian Dict",2001,8,0
"509","Fiji",950,950,"Melanesia","Oceania","Ratu Sir Kamisese Mara","Ratu Sir Kamisese Mara.Fiji.1970.1986.Civilian Dict","Non-democracy","Civilian Dict",1970,17,1
"510","Fiji",950,950,"Melanesia","Oceania","Sitiveni Rabuka","Sitiveni Rabuka.Fiji.1987.1991.Military Dict","Non-democracy","Military Dict",1987,5,1
"511","Fiji",950,950,"Melanesia","Oceania","Sitiveni Rabuka","Sitiveni Rabuka.Fiji.1992.1998.Parliamentary Dem","Democracy","Parliamentary Dem",1992,7,1
"512","Fiji",950,950,"Melanesia","Oceania","Mahendra Chaudhry","Mahendra Chaudhry.Fiji.1999.1999.Parliamentary Dem","Democracy","Parliamentary Dem",1999,1,1
"513","Fiji",950,950,"Melanesia","Oceania","Ratu Josefa Iloilo","Ratu Josefa Iloilo.Fiji.2000.2000.Military Dict","Non-democracy","Military Dict",2000,1,1
"514","Fiji",950,950,"Melanesia","Oceania","Ratu Josefa Iloilo","Ratu Josefa Iloilo.Fiji.2001.2005.Civilian Dict","Non-democracy","Civilian Dict",2001,5,1
"515","Fiji",950,950,"Melanesia","Oceania","Voreqe Bainimarama","Voreqe Bainimarama.Fiji.2006.2008.Military Dict","Non-democracy","Military Dict",2006,3,0
"516","Finland",375,375,"Northern Europe","Europe","Mauno Pekkala","Mauno Pekkala.Finland.1946.1947.Mixed Dem","Democracy","Mixed Dem",1946,2,1
"517","Finland",375,375,"Northern Europe","Europe","Karl Fagerholm","Karl Fagerholm.Finland.1948.1949.Mixed Dem","Democracy","Mixed Dem",1948,2,1
"518","Finland",375,375,"Northern Europe","Europe","Urho Kekkonen","Urho Kekkonen.Finland.1950.1952.Mixed Dem","Democracy","Mixed Dem",1950,3,1
"519","Finland",375,375,"Northern Europe","Europe","Sakari Tuomioja","Sakari Tuomioja.Finland.1953.1953.Mixed Dem","Democracy","Mixed Dem",1953,1,1
"520","Finland",375,375,"Northern Europe","Europe","Urho Kekkonen","Urho Kekkonen.Finland.1954.1955.Mixed Dem","Democracy","Mixed Dem",1954,2,1
"521","Finland",375,375,"Northern Europe","Europe","Karl Fagerholm","Karl Fagerholm.Finland.1956.1956.Mixed Dem","Democracy","Mixed Dem",1956,1,1
"522","Finland",375,375,"Northern Europe","Europe","Rainer von Fieandt","Rainer von Fieandt.Finland.1957.1957.Mixed Dem","Democracy","Mixed Dem",1957,1,1
"523","Finland",375,375,"Northern Europe","Europe","Karl Fagerholm","Karl Fagerholm.Finland.1958.1958.Mixed Dem","Democracy","Mixed Dem",1958,1,1
"524","Finland",375,375,"Northern Europe","Europe","Vaino Sukselainen","Vaino Sukselainen.Finland.1959.1960.Mixed Dem","Democracy","Mixed Dem",1959,2,1
"525","Finland",375,375,"Northern Europe","Europe","Martti Miettunen","Martti Miettunen.Finland.1961.1961.Mixed Dem","Democracy","Mixed Dem",1961,1,1
"526","Finland",375,375,"Northern Europe","Europe","Ahti Karjalainen","Ahti Karjalainen.Finland.1962.1962.Mixed Dem","Democracy","Mixed Dem",1962,1,1
"527","Finland",375,375,"Northern Europe","Europe","Reino Lehto","Reino Lehto.Finland.1963.1963.Mixed Dem","Democracy","Mixed Dem",1963,1,1
"528","Finland",375,375,"Northern Europe","Europe","Johannes Virolainen","Johannes Virolainen.Finland.1964.1965.Mixed Dem","Democracy","Mixed Dem",1964,2,1
"529","Finland",375,375,"Northern Europe","Europe","Rafael Paasio","Rafael Paasio.Finland.1966.1967.Mixed Dem","Democracy","Mixed Dem",1966,2,1
"530","Finland",375,375,"Northern Europe","Europe","Mauno Koivisto","Mauno Koivisto.Finland.1968.1969.Mixed Dem","Democracy","Mixed Dem",1968,2,1
"531","Finland",375,375,"Northern Europe","Europe","Ahti Karjalainen","Ahti Karjalainen.Finland.1970.1970.Mixed Dem","Democracy","Mixed Dem",1970,1,1
"532","Finland",375,375,"Northern Europe","Europe","Teuvo Aura","Teuvo Aura.Finland.1971.1971.Mixed Dem","Democracy","Mixed Dem",1971,1,1
"533","Finland",375,375,"Northern Europe","Europe","Kalevi Sorsa","Kalevi Sorsa.Finland.1972.1974.Mixed Dem","Democracy","Mixed Dem",1972,3,1
"534","Finland",375,375,"Northern Europe","Europe","Martti Miettunen","Martti Miettunen.Finland.1975.1976.Mixed Dem","Democracy","Mixed Dem",1975,2,1
"535","Finland",375,375,"Northern Europe","Europe","Kalevi Sorsa","Kalevi Sorsa.Finland.1977.1978.Mixed Dem","Democracy","Mixed Dem",1977,2,1
"536","Finland",375,375,"Northern Europe","Europe","Mauno Koivisto","Mauno Koivisto.Finland.1979.1981.Mixed Dem","Democracy","Mixed Dem",1979,3,1
"537","Finland",375,375,"Northern Europe","Europe","Kalevi Sorsa","Kalevi Sorsa.Finland.1982.1986.Mixed Dem","Democracy","Mixed Dem",1982,5,1
"538","Finland",375,375,"Northern Europe","Europe","Harri Holkeri","Harri Holkeri.Finland.1987.1990.Mixed Dem","Democracy","Mixed Dem",1987,4,1
"539","Finland",375,375,"Northern Europe","Europe","Esko Aho","Esko Aho.Finland.1991.1994.Mixed Dem","Democracy","Mixed Dem",1991,4,1
"540","Finland",375,375,"Northern Europe","Europe","Paavo Lipponen","Paavo Lipponen.Finland.1995.2002.Mixed Dem","Democracy","Mixed Dem",1995,8,1
"541","Finland",375,375,"Northern Europe","Europe","Matti Taneli Vanhanen","Matti Taneli Vanhanen.Finland.2003.2008.Mixed Dem","Democracy","Mixed Dem",2003,6,0
"542","France",220,220,"Western Europe","Europe","Leon Blum","Leon Blum.France.1946.1946.Parliamentary Dem","Democracy","Parliamentary Dem",1946,1,1
"543","France",220,220,"Western Europe","Europe","Robert Schuman","Robert Schuman.France.1947.1947.Parliamentary Dem","Democracy","Parliamentary Dem",1947,1,1
"544","France",220,220,"Western Europe","Europe","Henri Queuille","Henri Queuille.France.1948.1948.Parliamentary Dem","Democracy","Parliamentary Dem",1948,1,1
"545","France",220,220,"Western Europe","Europe","Georges Bidault","Georges Bidault.France.1949.1949.Parliamentary Dem","Democracy","Parliamentary Dem",1949,1,1
"546","France",220,220,"Western Europe","Europe","Rene Pleven","Rene Pleven.France.1950.1951.Parliamentary Dem","Democracy","Parliamentary Dem",1950,2,1
"547","France",220,220,"Western Europe","Europe","Antoine Pinay","Antoine Pinay.France.1952.1952.Parliamentary Dem","Democracy","Parliamentary Dem",1952,1,1
"548","France",220,220,"Western Europe","Europe","Joseph Laniel","Joseph Laniel.France.1953.1953.Parliamentary Dem","Democracy","Parliamentary Dem",1953,1,1
"549","France",220,220,"Western Europe","Europe","Pierre Mendes-France","Pierre Mendes-France.France.1954.1954.Parliamentary Dem","Democracy","Parliamentary Dem",1954,1,1
"550","France",220,220,"Western Europe","Europe","Edgar Faure","Edgar Faure.France.1955.1955.Parliamentary Dem","Democracy","Parliamentary Dem",1955,1,1
"551","France",220,220,"Western Europe","Europe","Guy Mollet","Guy Mollet.France.1956.1956.Parliamentary Dem","Democracy","Parliamentary Dem",1956,1,1
"552","France",220,220,"Western Europe","Europe","Felix Gaillard","Felix Gaillard.France.1957.1957.Parliamentary Dem","Democracy","Parliamentary Dem",1957,1,1
"553","France",220,220,"Western Europe","Europe","Charles de Gaulle","Charles de Gaulle.France.1958.1958.Parliamentary Dem","Democracy","Parliamentary Dem",1958,1,1
"554","France",220,220,"Western Europe","Europe","Michel Debre","Michel Debre.France.1959.1961.Parliamentary Dem","Democracy","Parliamentary Dem",1959,3,1
"555","France",220,220,"Western Europe","Europe","Georges Pompidou","Georges Pompidou.France.1962.1964.Parliamentary Dem","Democracy","Parliamentary Dem",1962,3,1
"556","France",220,220,"Western Europe","Europe","Georges Pompidou","Georges Pompidou.France.1965.1967.Mixed Dem","Democracy","Mixed Dem",1965,3,1
"557","France",220,220,"Western Europe","Europe","Maurice Couve de Murville","Maurice Couve de Murville.France.1968.1968.Mixed Dem","Democracy","Mixed Dem",1968,1,1
"558","France",220,220,"Western Europe","Europe","Jacques Chaban-Delmas","Jacques Chaban-Delmas.France.1969.1971.Mixed Dem","Democracy","Mixed Dem",1969,3,1
"559","France",220,220,"Western Europe","Europe","Pierre Messmer","Pierre Messmer.France.1972.1973.Mixed Dem","Democracy","Mixed Dem",1972,2,1
"560","France",220,220,"Western Europe","Europe","Jacques Chirac","Jacques Chirac.France.1974.1975.Mixed Dem","Democracy","Mixed Dem",1974,2,1
"561","France",220,220,"Western Europe","Europe","Raymond Barre","Raymond Barre.France.1976.1980.Mixed Dem","Democracy","Mixed Dem",1976,5,1
"562","France",220,220,"Western Europe","Europe","Pierre Mauroy","Pierre Mauroy.France.1981.1983.Mixed Dem","Democracy","Mixed Dem",1981,3,1
"563","France",220,220,"Western Europe","Europe","Laurent Fabius","Laurent Fabius.France.1984.1985.Mixed Dem","Democracy","Mixed Dem",1984,2,1
"564","France",220,220,"Western Europe","Europe","Jacques Chirac","Jacques Chirac.France.1986.1987.Mixed Dem","Democracy","Mixed Dem",1986,2,1
"565","France",220,220,"Western Europe","Europe","Michel Rocard","Michel Rocard.France.1988.1990.Mixed Dem","Democracy","Mixed Dem",1988,3,1
"566","France",220,220,"Western Europe","Europe","Edith Cresson","Edith Cresson.France.1991.1991.Mixed Dem","Democracy","Mixed Dem",1991,1,1
"567","France",220,220,"Western Europe","Europe","Pierre Beregovoy","Pierre Beregovoy.France.1992.1992.Mixed Dem","Democracy","Mixed Dem",1992,1,1
"568","France",220,220,"Western Europe","Europe","Edouard Balladur","Edouard Balladur.France.1993.1994.Mixed Dem","Democracy","Mixed Dem",1993,2,1
"569","France",220,220,"Western Europe","Europe","Alain Juppe","Alain Juppe.France.1995.1996.Mixed Dem","Democracy","Mixed Dem",1995,2,1
"570","France",220,220,"Western Europe","Europe","Lionel Jospin","Lionel Jospin.France.1997.2001.Mixed Dem","Democracy","Mixed Dem",1997,5,1
"571","France",220,220,"Western Europe","Europe","Jean-Pierre Raffarin","Jean-Pierre Raffarin.France.2002.2004.Mixed Dem","Democracy","Mixed Dem",2002,3,1
"572","France",220,220,"Western Europe","Europe","Dominique de Villepin","Dominique de Villepin.France.2005.2006.Mixed Dem","Democracy","Mixed Dem",2005,2,1
"573","France",220,220,"Western Europe","Europe","Franï¿½ois Fillon","Franï¿½ois Fillon.France.2007.2008.Mixed Dem","Democracy","Mixed Dem",2007,2,0
"574","Gabon",481,481,"Middle Africa","Africa","Leon Mba","Leon Mba.Gabon.1960.1966.Civilian Dict","Non-democracy","Civilian Dict",1960,7,0
"575","Gabon",481,481,"Middle Africa","Africa","Albert-Bernard Bongo","Albert-Bernard Bongo.Gabon.1967.2008.Civilian Dict","Non-democracy","Civilian Dict",1967,42,0
"576","Gambia",420,420,"Western Africa","Africa","Dauda Jawara","Dauda Jawara.Gambia.1965.1993.Civilian Dict","Non-democracy","Civilian Dict",1965,29,1
"577","Gambia",420,420,"Western Africa","Africa","Yahya Jammeh","Yahya Jammeh.Gambia.1994.2008.Military Dict","Non-democracy","Military Dict",1994,15,0
"578","Georgia",372,372,"Western Asia","Asia","Zviad Gamsakhurdia","Zviad Gamsakhurdia.Georgia.1991.1991.Civilian Dict","Non-democracy","Civilian Dict",1991,1,1
"579","Georgia",372,372,"Western Asia","Asia","Eduard Shevardnadze","Eduard Shevardnadze.Georgia.1992.2002.Civilian Dict","Non-democracy","Civilian Dict",1992,11,1
"580","Georgia",372,372,"Western Asia","Asia","Nino Burdzhanadze","Nino Burdzhanadze.Georgia.2003.2003.Civilian Dict","Non-democracy","Civilian Dict",2003,1,1
"581","Georgia",372,372,"Western Asia","Asia","Zurab Zhvania","Zurab Zhvania.Georgia.2004.2004.Mixed Dem","Democracy","Mixed Dem",2004,1,1
"582","Georgia",372,372,"Western Asia","Asia","Zurab Nogaideli","Zurab Nogaideli.Georgia.2005.2006.Mixed Dem","Democracy","Mixed Dem",2005,2,1
"583","Georgia",372,372,"Western Asia","Asia","Lado Gurgenidze","Lado Gurgenidze.Georgia.2007.2007.Mixed Dem","Democracy","Mixed Dem",2007,1,1
"584","Georgia",372,372,"Western Asia","Asia","Grigol Mgaloblishvili","Grigol Mgaloblishvili.Georgia.2008.2008.Mixed Dem","Democracy","Mixed Dem",2008,1,0
"585","Germany",255,255,"Western Europe","Europe","Helmut Kohl","Helmut Kohl.Germany.1990.1997.Parliamentary Dem","Democracy","Parliamentary Dem",1990,8,1
"586","Germany",255,255,"Western Europe","Europe","Gerhard Schroeder","Gerhard Schroeder.Germany.1998.2004.Parliamentary Dem","Democracy","Parliamentary Dem",1998,7,1
"587","Germany",255,255,"Western Europe","Europe","Angela Merkel","Angela Merkel.Germany.2005.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2005,4,0
"588","Germany, East",265,265,"Western Europe","Europe","Wilhelm Pieck + Otto Grotewohl","Wilhelm Pieck + Otto Grotewohl.Germany, East.1949.1949.Civilian Dict","Non-democracy","Civilian Dict",1949,1,1
"589","Germany, East",265,265,"Western Europe","Europe","Walter Ulbricht","Walter Ulbricht.Germany, East.1950.1970.Civilian Dict","Non-democracy","Civilian Dict",1950,21,1
"590","Germany, East",265,265,"Western Europe","Europe","Erich Honecker","Erich Honecker.Germany, East.1971.1988.Civilian Dict","Non-democracy","Civilian Dict",1971,18,1
"591","Germany, East",265,265,"Western Europe","Europe","Egon Krenz","Egon Krenz.Germany, East.1989.1989.Civilian Dict","Non-democracy","Civilian Dict",1989,1,0
"592","Germany, West",260,260,"Western Europe","Europe","Konrad Adenauer","Konrad Adenauer.Germany, West.1949.1962.Parliamentary Dem","Democracy","Parliamentary Dem",1949,14,1
"593","Germany, West",260,260,"Western Europe","Europe","Ludwig Erhard","Ludwig Erhard.Germany, West.1963.1965.Parliamentary Dem","Democracy","Parliamentary Dem",1963,3,1
"594","Germany, West",260,260,"Western Europe","Europe","Kurt-Georg Kiesinger","Kurt-Georg Kiesinger.Germany, West.1966.1968.Parliamentary Dem","Democracy","Parliamentary Dem",1966,3,1
"595","Germany, West",260,260,"Western Europe","Europe","Willy Brandt","Willy Brandt.Germany, West.1969.1973.Parliamentary Dem","Democracy","Parliamentary Dem",1969,5,1
"596","Germany, West",260,260,"Western Europe","Europe","Helmut Schmidt","Helmut Schmidt.Germany, West.1974.1981.Parliamentary Dem","Democracy","Parliamentary Dem",1974,8,1
"597","Germany, West",260,260,"Western Europe","Europe","Helmut Kohl","Helmut Kohl.Germany, West.1982.1989.Parliamentary Dem","Democracy","Parliamentary Dem",1982,8,0
"598","Ghana",452,452,"Western Africa","Africa","Kwame Nkrumah","Kwame Nkrumah.Ghana.1957.1965.Civilian Dict","Non-democracy","Civilian Dict",1957,9,1
"599","Ghana",452,452,"Western Africa","Africa","Joseph Ankrah","Joseph Ankrah.Ghana.1966.1968.Military Dict","Non-democracy","Military Dict",1966,3,1
"600","Ghana",452,452,"Western Africa","Africa","Kofi Busia","Kofi Busia.Ghana.1969.1971.Parliamentary Dem","Democracy","Parliamentary Dem",1969,3,1
"601","Ghana",452,452,"Western Africa","Africa","Ignatius Acheampong","Ignatius Acheampong.Ghana.1972.1977.Military Dict","Non-democracy","Military Dict",1972,6,1
"602","Ghana",452,452,"Western Africa","Africa","Frederick Akuffo","Frederick Akuffo.Ghana.1978.1978.Military Dict","Non-democracy","Military Dict",1978,1,1
"603","Ghana",452,452,"Western Africa","Africa","Hilla Limann","Hilla Limann.Ghana.1979.1980.Presidential Dem","Democracy","Presidential Dem",1979,2,1
"604","Ghana",452,452,"Western Africa","Africa","Jerry Rawlings","Jerry Rawlings.Ghana.1981.1992.Military Dict","Non-democracy","Military Dict",1981,12,1
"605","Ghana",452,452,"Western Africa","Africa","Jerry Rawlings","Jerry Rawlings.Ghana.1993.2000.Presidential Dem","Democracy","Presidential Dem",1993,8,1
"606","Ghana",452,452,"Western Africa","Africa","John Agyekum Kufuor","John Agyekum Kufuor.Ghana.2001.2008.Presidential Dem","Democracy","Presidential Dem",2001,8,0
"607","Greece",350,350,"Southern Europe","Europe","Constantine Tsaldaris","Constantine Tsaldaris.Greece.1946.1946.Parliamentary Dem","Democracy","Parliamentary Dem",1946,1,1
"608","Greece",350,350,"Southern Europe","Europe","Themistocles Sophoulis","Themistocles Sophoulis.Greece.1947.1948.Parliamentary Dem","Democracy","Parliamentary Dem",1947,2,1
"609","Greece",350,350,"Southern Europe","Europe","Alexandros Diomidis","Alexandros Diomidis.Greece.1949.1949.Parliamentary Dem","Democracy","Parliamentary Dem",1949,1,1
"610","Greece",350,350,"Southern Europe","Europe","Nikolas Plastiras","Nikolas Plastiras.Greece.1950.1951.Parliamentary Dem","Democracy","Parliamentary Dem",1950,2,1
"611","Greece",350,350,"Southern Europe","Europe","Alexandros Papagos","Alexandros Papagos.Greece.1952.1954.Parliamentary Dem","Democracy","Parliamentary Dem",1952,3,0
"612","Greece",350,350,"Southern Europe","Europe","Konstantinos Karamanlis","Konstantinos Karamanlis.Greece.1955.1962.Parliamentary Dem","Democracy","Parliamentary Dem",1955,8,1
"613","Greece",350,350,"Southern Europe","Europe","Ioannis Paraskevopoulos","Ioannis Paraskevopoulos.Greece.1963.1963.Parliamentary Dem","Democracy","Parliamentary Dem",1963,1,1
"614","Greece",350,350,"Southern Europe","Europe","Georgios Papandreou","Georgios Papandreou.Greece.1964.1964.Parliamentary Dem","Democracy","Parliamentary Dem",1964,1,1
"615","Greece",350,350,"Southern Europe","Europe","Stephanos Stephanopoulos","Stephanos Stephanopoulos.Greece.1965.1965.Parliamentary Dem","Democracy","Parliamentary Dem",1965,1,1
"616","Greece",350,350,"Southern Europe","Europe","Ioannis Paraskevopoulos","Ioannis Paraskevopoulos.Greece.1966.1966.Parliamentary Dem","Democracy","Parliamentary Dem",1966,1,1
"617","Greece",350,350,"Southern Europe","Europe","Georgios Papadopoulos","Georgios Papadopoulos.Greece.1967.1972.Military Dict","Non-democracy","Military Dict",1967,6,1
"618","Greece",350,350,"Southern Europe","Europe","Phaidon Gizikis","Phaidon Gizikis.Greece.1973.1973.Military Dict","Non-democracy","Military Dict",1973,1,1
"619","Greece",350,350,"Southern Europe","Europe","Konstantinos Karamanlis","Konstantinos Karamanlis.Greece.1974.1979.Parliamentary Dem","Democracy","Parliamentary Dem",1974,6,1
"620","Greece",350,350,"Southern Europe","Europe","Georgios Rallis","Georgios Rallis.Greece.1980.1980.Parliamentary Dem","Democracy","Parliamentary Dem",1980,1,1
"621","Greece",350,350,"Southern Europe","Europe","Andreas Papandreou","Andreas Papandreou.Greece.1981.1988.Parliamentary Dem","Democracy","Parliamentary Dem",1981,8,1
"622","Greece",350,350,"Southern Europe","Europe","Xenophon Zolotas","Xenophon Zolotas.Greece.1989.1989.Parliamentary Dem","Democracy","Parliamentary Dem",1989,1,1
"623","Greece",350,350,"Southern Europe","Europe","Konstantinos Mitsotakis","Konstantinos Mitsotakis.Greece.1990.1992.Parliamentary Dem","Democracy","Parliamentary Dem",1990,3,1
"624","Greece",350,350,"Southern Europe","Europe","Andreas Papandreou","Andreas Papandreou.Greece.1993.1995.Parliamentary Dem","Democracy","Parliamentary Dem",1993,3,0
"625","Greece",350,350,"Southern Europe","Europe","Kostas Simitis","Kostas Simitis.Greece.1996.2003.Parliamentary Dem","Democracy","Parliamentary Dem",1996,8,1
"626","Greece",350,350,"Southern Europe","Europe","Konstantinos A. (Kostas) Karamanlis","Konstantinos A. (Kostas) Karamanlis.Greece.2004.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2004,5,0
"627","Grenada",55,55,"Caribbean","Americas","Eric Gairy","Eric Gairy.Grenada.1974.1978.Parliamentary Dem","Democracy","Parliamentary Dem",1974,5,1
"628","Grenada",55,55,"Caribbean","Americas","Maurice Bishop","Maurice Bishop.Grenada.1979.1982.Civilian Dict","Non-democracy","Civilian Dict",1979,4,0
"629","Grenada",55,55,"Caribbean","Americas","Nicholas Brathwaite","Nicholas Brathwaite.Grenada.1983.1983.Civilian Dict","Non-democracy","Civilian Dict",1983,1,1
"630","Grenada",55,55,"Caribbean","Americas","Herbert Blaize","Herbert Blaize.Grenada.1984.1988.Parliamentary Dem","Democracy","Parliamentary Dem",1984,5,0
"631","Grenada",55,55,"Caribbean","Americas","Ben Jones","Ben Jones.Grenada.1989.1989.Parliamentary Dem","Democracy","Parliamentary Dem",1989,1,1
"632","Grenada",55,55,"Caribbean","Americas","Nicholas Brathwaite","Nicholas Brathwaite.Grenada.1990.1994.Parliamentary Dem","Democracy","Parliamentary Dem",1990,5,1
"633","Grenada",55,55,"Caribbean","Americas","Keith Mitchell","Keith Mitchell.Grenada.1995.2007.Parliamentary Dem","Democracy","Parliamentary Dem",1995,13,1
"634","Grenada",55,55,"Caribbean","Americas","Tillman Joseph Thomas","Tillman Joseph Thomas.Grenada.2008.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2008,1,0
"635","Guatemala",90,90,"Central America","Americas","Juan Arevalo","Juan Arevalo.Guatemala.1946.1950.Presidential Dem","Democracy","Presidential Dem",1946,5,1
"636","Guatemala",90,90,"Central America","Americas","Jacobo Arbenz Guzman","Jacobo Arbenz Guzman.Guatemala.1951.1953.Presidential Dem","Democracy","Presidential Dem",1951,3,1
"637","Guatemala",90,90,"Central America","Americas","Carlos Castillo Armas","Carlos Castillo Armas.Guatemala.1954.1956.Military Dict","Non-democracy","Military Dict",1954,3,0
"638","Guatemala",90,90,"Central America","Americas","Guillermo Flores Avendano","Guillermo Flores Avendano.Guatemala.1957.1957.Military Dict","Non-democracy","Military Dict",1957,1,1
"639","Guatemala",90,90,"Central America","Americas","Miguel Ydigoras Fuentes","Miguel Ydigoras Fuentes.Guatemala.1958.1962.Presidential Dem","Democracy","Presidential Dem",1958,5,1
"640","Guatemala",90,90,"Central America","Americas","Enrique Peralta Azurdia","Enrique Peralta Azurdia.Guatemala.1963.1965.Military Dict","Non-democracy","Military Dict",1963,3,1
"641","Guatemala",90,90,"Central America","Americas","Julio Mendez Montenegro","Julio Mendez Montenegro.Guatemala.1966.1969.Presidential Dem","Democracy","Presidential Dem",1966,4,1
"642","Guatemala",90,90,"Central America","Americas","Carlos Arana Osorio","Carlos Arana Osorio.Guatemala.1970.1973.Presidential Dem","Democracy","Presidential Dem",1970,4,1
"643","Guatemala",90,90,"Central America","Americas","Kjell Laugerud Garcia","Kjell Laugerud Garcia.Guatemala.1974.1977.Presidential Dem","Democracy","Presidential Dem",1974,4,1
"644","Guatemala",90,90,"Central America","Americas","Romeo Lucas Garcia","Romeo Lucas Garcia.Guatemala.1978.1981.Presidential Dem","Democracy","Presidential Dem",1978,4,1
"645","Guatemala",90,90,"Central America","Americas","Efrain Rios Montt","Efrain Rios Montt.Guatemala.1982.1982.Military Dict","Non-democracy","Military Dict",1982,1,1
"646","Guatemala",90,90,"Central America","Americas","Oscar Humberto Mejia Victores","Oscar Humberto Mejia Victores.Guatemala.1983.1985.Military Dict","Non-democracy","Military Dict",1983,3,1
"647","Guatemala",90,90,"Central America","Americas","Marco Vinicio Cerezo Arevalo","Marco Vinicio Cerezo Arevalo.Guatemala.1986.1990.Presidential Dem","Democracy","Presidential Dem",1986,5,1
"648","Guatemala",90,90,"Central America","Americas","Jorge Serrano Elias","Jorge Serrano Elias.Guatemala.1991.1992.Presidential Dem","Democracy","Presidential Dem",1991,2,1
"649","Guatemala",90,90,"Central America","Americas","Ramiro de Leon Carpio","Ramiro de Leon Carpio.Guatemala.1993.1995.Presidential Dem","Democracy","Presidential Dem",1993,3,1
"650","Guatemala",90,90,"Central America","Americas","Alvaro Arzu Irigoyen","Alvaro Arzu Irigoyen.Guatemala.1996.1999.Presidential Dem","Democracy","Presidential Dem",1996,4,1
"651","Guatemala",90,90,"Central America","Americas","Alfonso Antonio Portillo Cabrera","Alfonso Antonio Portillo Cabrera.Guatemala.2000.2007.Presidential Dem","Democracy","Presidential Dem",2000,8,1
"652","Guatemala",90,90,"Central America","Americas","ï¿½lvaro Colom Caballeros","ï¿½lvaro Colom Caballeros.Guatemala.2008.2008.Presidential Dem","Democracy","Presidential Dem",2008,1,0
"653","Guinea",438,438,"Western Africa","Africa","Ahmed Sekou Toure","Ahmed Sekou Toure.Guinea.1958.1983.Civilian Dict","Non-democracy","Civilian Dict",1958,26,0
"654","Guinea",438,438,"Western Africa","Africa","Lansana Conte","Lansana Conte.Guinea.1984.2007.Military Dict","Non-democracy","Military Dict",1984,24,0
"655","Guinea",438,438,"Western Africa","Africa","Moussa Dadis Camara","Moussa Dadis Camara.Guinea.2008.2008.Military Dict","Non-democracy","Military Dict",2008,1,0
"656","Guinea-Bissau",404,404,"Western Africa","Africa","Luis Cabral","Luis Cabral.Guinea-Bissau.1974.1979.Civilian Dict","Non-democracy","Civilian Dict",1974,6,1
"657","Guinea-Bissau",404,404,"Western Africa","Africa","Joao Vieira","Joao Vieira.Guinea-Bissau.1980.1998.Military Dict","Non-democracy","Military Dict",1980,19,1
"658","Guinea-Bissau",404,404,"Western Africa","Africa","Ansumane Mane","Ansumane Mane.Guinea-Bissau.1999.1999.Military Dict","Non-democracy","Military Dict",1999,1,1
"659","Guinea-Bissau",404,404,"Western Africa","Africa","Kumba Iala","Kumba Iala.Guinea-Bissau.2000.2002.Mixed Dem","Democracy","Mixed Dem",2000,3,1
"660","Guinea-Bissau",404,404,"Western Africa","Africa","Henrique Rosa","Henrique Rosa.Guinea-Bissau.2003.2003.Civilian Dict","Non-democracy","Civilian Dict",2003,1,1
"661","Guinea-Bissau",404,404,"Western Africa","Africa","Carlos Gomes","Carlos Gomes.Guinea-Bissau.2004.2004.Mixed Dem","Democracy","Mixed Dem",2004,1,1
"662","Guinea-Bissau",404,404,"Western Africa","Africa","Aristides Gomes","Aristides Gomes.Guinea-Bissau.2005.2006.Mixed Dem","Democracy","Mixed Dem",2005,2,1
"663","Guinea-Bissau",404,404,"Western Africa","Africa","Martinho Ndafa Cabi","Martinho Ndafa Cabi.Guinea-Bissau.2007.2007.Mixed Dem","Democracy","Mixed Dem",2007,1,1
"664","Guinea-Bissau",404,404,"Western Africa","Africa","Carlos Correia","Carlos Correia.Guinea-Bissau.2008.2008.Mixed Dem","Democracy","Mixed Dem",2008,1,0
"665","Guyana",110,110,"South America","Americas","Forbes Burnham","Forbes Burnham.Guyana.1966.1984.Civilian Dict","Non-democracy","Civilian Dict",1966,19,0
"666","Guyana",110,110,"South America","Americas","Desmond Hoyte","Desmond Hoyte.Guyana.1985.1991.Civilian Dict","Non-democracy","Civilian Dict",1985,7,1
"667","Guyana",110,110,"South America","Americas","Cheddi Jagan","Cheddi Jagan.Guyana.1992.1996.Civilian Dict","Non-democracy","Civilian Dict",1992,5,0
"668","Guyana",110,110,"South America","Americas","Janet Jagan","Janet Jagan.Guyana.1997.1998.Civilian Dict","Non-democracy","Civilian Dict",1997,2,1
"669","Guyana",110,110,"South America","Americas","Bharrat Jagdeo","Bharrat Jagdeo.Guyana.1999.2008.Civilian Dict","Non-democracy","Civilian Dict",1999,10,0
"670","Haiti",41,41,"Caribbean","Americas","Dumarsais Estime","Dumarsais Estime.Haiti.1946.1949.Civilian Dict","Non-democracy","Civilian Dict",1946,4,1
"671","Haiti",41,41,"Caribbean","Americas","Paul Eugene Magloire","Paul Eugene Magloire.Haiti.1950.1955.Military Dict","Non-democracy","Military Dict",1950,6,1
"672","Haiti",41,41,"Caribbean","Americas","Joseph Nemours Pierre-Louis","Joseph Nemours Pierre-Louis.Haiti.1956.1956.Civilian Dict","Non-democracy","Civilian Dict",1956,1,1
"673","Haiti",41,41,"Caribbean","Americas","Francois Duvalier","Francois Duvalier.Haiti.1957.1970.Civilian Dict","Non-democracy","Civilian Dict",1957,14,0
"674","Haiti",41,41,"Caribbean","Americas","Jean-Claude Duvalier","Jean-Claude Duvalier.Haiti.1971.1985.Civilian Dict","Non-democracy","Civilian Dict",1971,15,1
"675","Haiti",41,41,"Caribbean","Americas","Henri Namphy","Henri Namphy.Haiti.1986.1987.Military Dict","Non-democracy","Military Dict",1986,2,1
"676","Haiti",41,41,"Caribbean","Americas","Prosper Avril","Prosper Avril.Haiti.1988.1989.Military Dict","Non-democracy","Military Dict",1988,2,1
"677","Haiti",41,41,"Caribbean","Americas","Ertha Pascal-Trouillot","Ertha Pascal-Trouillot.Haiti.1990.1990.Civilian Dict","Non-democracy","Civilian Dict",1990,1,1
"678","Haiti",41,41,"Caribbean","Americas","Joseph Nerette","Joseph Nerette.Haiti.1991.1991.Civilian Dict","Non-democracy","Civilian Dict",1991,1,1
"679","Haiti",41,41,"Caribbean","Americas","Marc Bazin","Marc Bazin.Haiti.1992.1992.Civilian Dict","Non-democracy","Civilian Dict",1992,1,1
"680","Haiti",41,41,"Caribbean","Americas","Jean-Bertrand Aristide","Jean-Bertrand Aristide.Haiti.1993.1995.Civilian Dict","Non-democracy","Civilian Dict",1993,3,1
"681","Haiti",41,41,"Caribbean","Americas","Rene Garcia Preval","Rene Garcia Preval.Haiti.1996.2000.Civilian Dict","Non-democracy","Civilian Dict",1996,5,1
"682","Haiti",41,41,"Caribbean","Americas","Jean-Bertrand Aristide","Jean-Bertrand Aristide.Haiti.2001.2003.Civilian Dict","Non-democracy","Civilian Dict",2001,3,1
"683","Haiti",41,41,"Caribbean","Americas","Boniface Alexandre","Boniface Alexandre.Haiti.2004.2005.Civilian Dict","Non-democracy","Civilian Dict",2004,2,1
"684","Haiti",41,41,"Caribbean","Americas","Rene Garcia Preval","Rene Garcia Preval.Haiti.2006.2008.Civilian Dict","Non-democracy","Civilian Dict",2006,3,0
"685","Honduras",91,91,"Central America","Americas","Tiburcio Carias Andino","Tiburcio Carias Andino.Honduras.1946.1948.Military Dict","Non-democracy","Military Dict",1946,3,1
"686","Honduras",91,91,"Central America","Americas","Juan Manuel Galvez","Juan Manuel Galvez.Honduras.1949.1953.Civilian Dict","Non-democracy","Civilian Dict",1949,5,1
"687","Honduras",91,91,"Central America","Americas","Julio Lozano Diaz","Julio Lozano Diaz.Honduras.1954.1955.Civilian Dict","Non-democracy","Civilian Dict",1954,2,1
"688","Honduras",91,91,"Central America","Americas","Roque Rodriguez","Roque Rodriguez.Honduras.1956.1956.Military Dict","Non-democracy","Military Dict",1956,1,1
"689","Honduras",91,91,"Central America","Americas","Ramon Villeda Morales","Ramon Villeda Morales.Honduras.1957.1962.Presidential Dem","Democracy","Presidential Dem",1957,6,1
"690","Honduras",91,91,"Central America","Americas","Oswaldo Lopez Arellano","Oswaldo Lopez Arellano.Honduras.1963.1970.Military Dict","Non-democracy","Military Dict",1963,8,1
"691","Honduras",91,91,"Central America","Americas","Ramon Ernesto Cruz Ucles","Ramon Ernesto Cruz Ucles.Honduras.1971.1971.Presidential Dem","Democracy","Presidential Dem",1971,1,1
"692","Honduras",91,91,"Central America","Americas","Oswaldo Lopez Arellano","Oswaldo Lopez Arellano.Honduras.1972.1974.Military Dict","Non-democracy","Military Dict",1972,3,1
"693","Honduras",91,91,"Central America","Americas","Juan Alberto Melgar Castro","Juan Alberto Melgar Castro.Honduras.1975.1977.Military Dict","Non-democracy","Military Dict",1975,3,1
"694","Honduras",91,91,"Central America","Americas","Policarpo Paz Garcia","Policarpo Paz Garcia.Honduras.1978.1981.Military Dict","Non-democracy","Military Dict",1978,4,1
"695","Honduras",91,91,"Central America","Americas","Roberto Suazo Cordova","Roberto Suazo Cordova.Honduras.1982.1985.Presidential Dem","Democracy","Presidential Dem",1982,4,1
"696","Honduras",91,91,"Central America","Americas","Jose Azcona Hoyo","Jose Azcona Hoyo.Honduras.1986.1989.Presidential Dem","Democracy","Presidential Dem",1986,4,1
"697","Honduras",91,91,"Central America","Americas","Rafael Leonardo Callejas","Rafael Leonardo Callejas.Honduras.1990.1993.Presidential Dem","Democracy","Presidential Dem",1990,4,1
"698","Honduras",91,91,"Central America","Americas","Carlos Roberto Reina Idiaquez","Carlos Roberto Reina Idiaquez.Honduras.1994.1997.Presidential Dem","Democracy","Presidential Dem",1994,4,1
"699","Honduras",91,91,"Central America","Americas","Carlos Roberto Flores Facusse","Carlos Roberto Flores Facusse.Honduras.1998.2001.Presidential Dem","Democracy","Presidential Dem",1998,4,1
"700","Honduras",91,91,"Central America","Americas","Ricardo Maduro Joest","Ricardo Maduro Joest.Honduras.2002.2005.Presidential Dem","Democracy","Presidential Dem",2002,4,1
"701","Honduras",91,91,"Central America","Americas","Josï¿½ Manuel Zelaya Rosales","Josï¿½ Manuel Zelaya Rosales.Honduras.2006.2008.Presidential Dem","Democracy","Presidential Dem",2006,3,0
"702","Hungary",310,310,"Eastern Europe","Europe","Matyas Rakosi","Matyas Rakosi.Hungary.1946.1955.Civilian Dict","Non-democracy","Civilian Dict",1946,10,1
"703","Hungary",310,310,"Eastern Europe","Europe","Janos Kadar","Janos Kadar.Hungary.1956.1987.Civilian Dict","Non-democracy","Civilian Dict",1956,32,1
"704","Hungary",310,310,"Eastern Europe","Europe","Karoly Grosz","Karoly Grosz.Hungary.1988.1988.Civilian Dict","Non-democracy","Civilian Dict",1988,1,1
"705","Hungary",310,310,"Eastern Europe","Europe","Miklos Nemeth","Miklos Nemeth.Hungary.1989.1989.Civilian Dict","Non-democracy","Civilian Dict",1989,1,1
"706","Hungary",310,310,"Eastern Europe","Europe","Jozsef Antall","Jozsef Antall.Hungary.1990.1992.Parliamentary Dem","Democracy","Parliamentary Dem",1990,3,0
"707","Hungary",310,310,"Eastern Europe","Europe","Peter Boross","Peter Boross.Hungary.1993.1993.Parliamentary Dem","Democracy","Parliamentary Dem",1993,1,1
"708","Hungary",310,310,"Eastern Europe","Europe","Gyula Horn","Gyula Horn.Hungary.1994.1997.Parliamentary Dem","Democracy","Parliamentary Dem",1994,4,1
"709","Hungary",310,310,"Eastern Europe","Europe","Viktor Orban","Viktor Orban.Hungary.1998.2001.Parliamentary Dem","Democracy","Parliamentary Dem",1998,4,1
"710","Hungary",310,310,"Eastern Europe","Europe","Peter Medgyessy","Peter Medgyessy.Hungary.2002.2003.Parliamentary Dem","Democracy","Parliamentary Dem",2002,2,1
"711","Hungary",310,310,"Eastern Europe","Europe","Ferenc Gyurcsï¿½ny","Ferenc Gyurcsï¿½ny.Hungary.2004.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2004,5,0
"712","Iceland",395,395,"Northern Europe","Europe","Olafur Thors","Olafur Thors.Iceland.1946.1946.Mixed Dem","Democracy","Mixed Dem",1946,1,1
"713","Iceland",395,395,"Northern Europe","Europe","Stefan Stefansson","Stefan Stefansson.Iceland.1947.1948.Mixed Dem","Democracy","Mixed Dem",1947,2,1
"714","Iceland",395,395,"Northern Europe","Europe","Olafur Thors","Olafur Thors.Iceland.1949.1949.Mixed Dem","Democracy","Mixed Dem",1949,1,1
"715","Iceland",395,395,"Northern Europe","Europe","Steingrimur Steinthorsson","Steingrimur Steinthorsson.Iceland.1950.1952.Mixed Dem","Democracy","Mixed Dem",1950,3,1
"716","Iceland",395,395,"Northern Europe","Europe","Olafur Thors","Olafur Thors.Iceland.1953.1955.Mixed Dem","Democracy","Mixed Dem",1953,3,1
"717","Iceland",395,395,"Northern Europe","Europe","Hermann Jonasson","Hermann Jonasson.Iceland.1956.1957.Mixed Dem","Democracy","Mixed Dem",1956,2,1
"718","Iceland",395,395,"Northern Europe","Europe","Emil Jonsson","Emil Jonsson.Iceland.1958.1958.Mixed Dem","Democracy","Mixed Dem",1958,1,1
"719","Iceland",395,395,"Northern Europe","Europe","Olafur Thors","Olafur Thors.Iceland.1959.1962.Mixed Dem","Democracy","Mixed Dem",1959,4,1
"720","Iceland",395,395,"Northern Europe","Europe","Bjarni Benediktsson","Bjarni Benediktsson.Iceland.1963.1969.Mixed Dem","Democracy","Mixed Dem",1963,7,0
"721","Iceland",395,395,"Northern Europe","Europe","Johann Hafstein","Johann Hafstein.Iceland.1970.1970.Mixed Dem","Democracy","Mixed Dem",1970,1,1
"722","Iceland",395,395,"Northern Europe","Europe","Olafur Johannesson","Olafur Johannesson.Iceland.1971.1973.Mixed Dem","Democracy","Mixed Dem",1971,3,1
"723","Iceland",395,395,"Northern Europe","Europe","Geir Hallgrimsson","Geir Hallgrimsson.Iceland.1974.1977.Mixed Dem","Democracy","Mixed Dem",1974,4,1
"724","Iceland",395,395,"Northern Europe","Europe","Olafur Johannesson","Olafur Johannesson.Iceland.1978.1978.Mixed Dem","Democracy","Mixed Dem",1978,1,1
"725","Iceland",395,395,"Northern Europe","Europe","Benedikt Groendal","Benedikt Groendal.Iceland.1979.1979.Mixed Dem","Democracy","Mixed Dem",1979,1,1
"726","Iceland",395,395,"Northern Europe","Europe","Gunnar Thoroddsen","Gunnar Thoroddsen.Iceland.1980.1982.Mixed Dem","Democracy","Mixed Dem",1980,3,1
"727","Iceland",395,395,"Northern Europe","Europe","Steingrimur Hermannsson","Steingrimur Hermannsson.Iceland.1983.1986.Mixed Dem","Democracy","Mixed Dem",1983,4,1
"728","Iceland",395,395,"Northern Europe","Europe","Thorsteinn Palsson","Thorsteinn Palsson.Iceland.1987.1987.Mixed Dem","Democracy","Mixed Dem",1987,1,1
"729","Iceland",395,395,"Northern Europe","Europe","Steingrimur Hermannsson","Steingrimur Hermannsson.Iceland.1988.1990.Mixed Dem","Democracy","Mixed Dem",1988,3,1
"730","Iceland",395,395,"Northern Europe","Europe","David Oddsson","David Oddsson.Iceland.1991.2003.Mixed Dem","Democracy","Mixed Dem",1991,13,1
"731","Iceland",395,395,"Northern Europe","Europe","Halldï¿½r ï¿½sgrï¿½msson","Halldï¿½r ï¿½sgrï¿½msson.Iceland.2004.2005.Mixed Dem","Democracy","Mixed Dem",2004,2,1
"732","Iceland",395,395,"Northern Europe","Europe","Geir Hilmar Haarde","Geir Hilmar Haarde.Iceland.2006.2008.Mixed Dem","Democracy","Mixed Dem",2006,3,0
"733","India",750,750,"Southern Asia","Asia","Jawaharlal Nehru","Jawaharlal Nehru.India.1947.1963.Parliamentary Dem","Democracy","Parliamentary Dem",1947,17,0
"734","India",750,750,"Southern Asia","Asia","Lal Bahadur Shastri","Lal Bahadur Shastri.India.1964.1965.Parliamentary Dem","Democracy","Parliamentary Dem",1964,2,1
"735","India",750,750,"Southern Asia","Asia","Indira Gandhi","Indira Gandhi.India.1966.1976.Parliamentary Dem","Democracy","Parliamentary Dem",1966,11,1
"736","India",750,750,"Southern Asia","Asia","Morarji Desai","Morarji Desai.India.1977.1978.Parliamentary Dem","Democracy","Parliamentary Dem",1977,2,1
"737","India",750,750,"Southern Asia","Asia","Charan Singh","Charan Singh.India.1979.1979.Parliamentary Dem","Democracy","Parliamentary Dem",1979,1,1
"738","India",750,750,"Southern Asia","Asia","Indira Gandhi","Indira Gandhi.India.1980.1983.Parliamentary Dem","Democracy","Parliamentary Dem",1980,4,0
"739","India",750,750,"Southern Asia","Asia","Rajiv Gandhi","Rajiv Gandhi.India.1984.1988.Parliamentary Dem","Democracy","Parliamentary Dem",1984,5,0
"740","India",750,750,"Southern Asia","Asia","Vishwanath Pratap Singh","Vishwanath Pratap Singh.India.1989.1989.Parliamentary Dem","Democracy","Parliamentary Dem",1989,1,1
"741","India",750,750,"Southern Asia","Asia","Chandra Shekhar","Chandra Shekhar.India.1990.1990.Parliamentary Dem","Democracy","Parliamentary Dem",1990,1,1
"742","India",750,750,"Southern Asia","Asia","P.V. Narasimha Rao","P.V. Narasimha Rao.India.1991.1995.Parliamentary Dem","Democracy","Parliamentary Dem",1991,5,1
"743","India",750,750,"Southern Asia","Asia","H.D. Deve Gowda","H.D. Deve Gowda.India.1996.1996.Parliamentary Dem","Democracy","Parliamentary Dem",1996,1,1
"744","India",750,750,"Southern Asia","Asia","Inder Kumar Gujral","Inder Kumar Gujral.India.1997.1997.Parliamentary Dem","Democracy","Parliamentary Dem",1997,1,1
"745","India",750,750,"Southern Asia","Asia","Atal Bihari Vajpayee","Atal Bihari Vajpayee.India.1998.2003.Parliamentary Dem","Democracy","Parliamentary Dem",1998,6,1
"746","India",750,750,"Southern Asia","Asia","Manmohan Singh","Manmohan Singh.India.2004.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2004,5,0
"747","Indonesia",850,850,"South-Eastern Asia","Asia","Sukarno","Sukarno.Indonesia.1949.1965.Civilian Dict","Non-democracy","Civilian Dict",1949,17,1
"748","Indonesia",850,850,"South-Eastern Asia","Asia","Suharto","Suharto.Indonesia.1966.1997.Military Dict","Non-democracy","Military Dict",1966,32,1
"749","Indonesia",850,850,"South-Eastern Asia","Asia","military","military.Indonesia.1998.1998.Military Dict","Non-democracy","Military Dict",1998,1,1
"750","Indonesia",850,850,"South-Eastern Asia","Asia","Abdurrahman Wahid","Abdurrahman Wahid.Indonesia.1999.2000.Presidential Dem","Democracy","Presidential Dem",1999,2,1
"751","Indonesia",850,850,"South-Eastern Asia","Asia","Megawati Sukarnoputri","Megawati Sukarnoputri.Indonesia.2001.2003.Presidential Dem","Democracy","Presidential Dem",2001,3,1
"752","Indonesia",850,850,"South-Eastern Asia","Asia","Susilo Bambang Yudhoyono","Susilo Bambang Yudhoyono.Indonesia.2004.2008.Presidential Dem","Democracy","Presidential Dem",2004,5,0
"753","Iran",630,630,"Southern Asia","Asia","Mohammed Reza Pahlavi","Mohammed Reza Pahlavi.Iran.1946.1978.Monarchy","Non-democracy","Monarchy",1946,33,1
"754","Iran",630,630,"Southern Asia","Asia","Ayatollah Ruhollah Khomeini","Ayatollah Ruhollah Khomeini.Iran.1979.1988.Civilian Dict","Non-democracy","Civilian Dict",1979,10,0
"755","Iran",630,630,"Southern Asia","Asia","Ayatollah Sayyed Ali Khamenei","Ayatollah Sayyed Ali Khamenei.Iran.1989.2008.Civilian Dict","Non-democracy","Civilian Dict",1989,20,0
"756","Iraq",645,645,"Western Asia","Asia","Abdul-Ilah","Abdul-Ilah.Iraq.1946.1952.Monarchy","Non-democracy","Monarchy",1946,7,1
"757","Iraq",645,645,"Western Asia","Asia","Faisal II","Faisal II.Iraq.1953.1957.Monarchy","Non-democracy","Monarchy",1953,5,0
"758","Iraq",645,645,"Western Asia","Asia","Najib el-Rubai","Najib el-Rubai.Iraq.1958.1962.Military Dict","Non-democracy","Military Dict",1958,5,1
"759","Iraq",645,645,"Western Asia","Asia","Abdul Salam Arif","Abdul Salam Arif.Iraq.1963.1965.Military Dict","Non-democracy","Military Dict",1963,3,0
"760","Iraq",645,645,"Western Asia","Asia","Abdul Rahman Arif","Abdul Rahman Arif.Iraq.1966.1967.Military Dict","Non-democracy","Military Dict",1966,2,1
"761","Iraq",645,645,"Western Asia","Asia","Ahmed al-Bakr","Ahmed al-Bakr.Iraq.1968.1978.Military Dict","Non-democracy","Military Dict",1968,11,1
"762","Iraq",645,645,"Western Asia","Asia","Saddam Hussein","Saddam Hussein.Iraq.1979.2002.Civilian Dict","Non-democracy","Civilian Dict",1979,24,1
"763","Iraq",645,645,"Western Asia","Asia","Lt. Gen. John P. Abizaid","Lt. Gen. John P. Abizaid.Iraq.2003.2008.Military Dict","Non-democracy","Military Dict",2003,6,0
"764","Ireland",205,205,"Northern Europe","Europe","Eamon de Valera","Eamon de Valera.Ireland.1946.1947.Mixed Dem","Democracy","Mixed Dem",1946,2,1
"765","Ireland",205,205,"Northern Europe","Europe","John Costello","John Costello.Ireland.1948.1950.Mixed Dem","Democracy","Mixed Dem",1948,3,1
"766","Ireland",205,205,"Northern Europe","Europe","Eamon de Valera","Eamon de Valera.Ireland.1951.1953.Mixed Dem","Democracy","Mixed Dem",1951,3,1
"767","Ireland",205,205,"Northern Europe","Europe","John Costello","John Costello.Ireland.1954.1956.Mixed Dem","Democracy","Mixed Dem",1954,3,1
"768","Ireland",205,205,"Northern Europe","Europe","Eamon de Valera","Eamon de Valera.Ireland.1957.1958.Mixed Dem","Democracy","Mixed Dem",1957,2,1
"769","Ireland",205,205,"Northern Europe","Europe","Sean Lemass","Sean Lemass.Ireland.1959.1965.Mixed Dem","Democracy","Mixed Dem",1959,7,1
"770","Ireland",205,205,"Northern Europe","Europe","John Lynch","John Lynch.Ireland.1966.1972.Mixed Dem","Democracy","Mixed Dem",1966,7,1
"771","Ireland",205,205,"Northern Europe","Europe","Liam Cosgrave","Liam Cosgrave.Ireland.1973.1976.Mixed Dem","Democracy","Mixed Dem",1973,4,1
"772","Ireland",205,205,"Northern Europe","Europe","John Lynch","John Lynch.Ireland.1977.1978.Mixed Dem","Democracy","Mixed Dem",1977,2,1
"773","Ireland",205,205,"Northern Europe","Europe","Charles Haughey","Charles Haughey.Ireland.1979.1980.Mixed Dem","Democracy","Mixed Dem",1979,2,1
"774","Ireland",205,205,"Northern Europe","Europe","Garret Fitzgerald","Garret Fitzgerald.Ireland.1981.1986.Mixed Dem","Democracy","Mixed Dem",1981,6,1
"775","Ireland",205,205,"Northern Europe","Europe","Charles Haughey","Charles Haughey.Ireland.1987.1991.Mixed Dem","Democracy","Mixed Dem",1987,5,1
"776","Ireland",205,205,"Northern Europe","Europe","Albert Reynolds","Albert Reynolds.Ireland.1992.1993.Mixed Dem","Democracy","Mixed Dem",1992,2,1
"777","Ireland",205,205,"Northern Europe","Europe","John Bruton","John Bruton.Ireland.1994.1996.Mixed Dem","Democracy","Mixed Dem",1994,3,1
"778","Ireland",205,205,"Northern Europe","Europe","Bertie Ahern","Bertie Ahern.Ireland.1997.2007.Mixed Dem","Democracy","Mixed Dem",1997,11,1
"779","Ireland",205,205,"Northern Europe","Europe","Brian Cowen","Brian Cowen.Ireland.2008.2008.Mixed Dem","Democracy","Mixed Dem",2008,1,0
"780","Israel",666,666,"Western Asia","Asia","David Ben-Gurion","David Ben-Gurion.Israel.1948.1952.Parliamentary Dem","Democracy","Parliamentary Dem",1948,5,1
"781","Israel",666,666,"Western Asia","Asia","Moshe Sharett","Moshe Sharett.Israel.1953.1954.Parliamentary Dem","Democracy","Parliamentary Dem",1953,2,1
"782","Israel",666,666,"Western Asia","Asia","David Ben-Gurion","David Ben-Gurion.Israel.1955.1962.Parliamentary Dem","Democracy","Parliamentary Dem",1955,8,1
"783","Israel",666,666,"Western Asia","Asia","Levi Eshkol","Levi Eshkol.Israel.1963.1968.Parliamentary Dem","Democracy","Parliamentary Dem",1963,6,0
"784","Israel",666,666,"Western Asia","Asia","Golda Meir","Golda Meir.Israel.1969.1973.Parliamentary Dem","Democracy","Parliamentary Dem",1969,5,1
"785","Israel",666,666,"Western Asia","Asia","Yitzhak Rabin","Yitzhak Rabin.Israel.1974.1976.Parliamentary Dem","Democracy","Parliamentary Dem",1974,3,1
"786","Israel",666,666,"Western Asia","Asia","Menachem Begin","Menachem Begin.Israel.1977.1982.Parliamentary Dem","Democracy","Parliamentary Dem",1977,6,1
"787","Israel",666,666,"Western Asia","Asia","Yitzhak Shamir","Yitzhak Shamir.Israel.1983.1983.Parliamentary Dem","Democracy","Parliamentary Dem",1983,1,1
"788","Israel",666,666,"Western Asia","Asia","Shimon Peres","Shimon Peres.Israel.1984.1985.Parliamentary Dem","Democracy","Parliamentary Dem",1984,2,1
"789","Israel",666,666,"Western Asia","Asia","Yitzhak Shamir","Yitzhak Shamir.Israel.1986.1991.Parliamentary Dem","Democracy","Parliamentary Dem",1986,6,1
"790","Israel",666,666,"Western Asia","Asia","Yitzhak Rabin","Yitzhak Rabin.Israel.1992.1994.Parliamentary Dem","Democracy","Parliamentary Dem",1992,3,0
"791","Israel",666,666,"Western Asia","Asia","Shimon Peres","Shimon Peres.Israel.1995.1995.Parliamentary Dem","Democracy","Parliamentary Dem",1995,1,1
"792","Israel",666,666,"Western Asia","Asia","Benjamin Netanyahu","Benjamin Netanyahu.Israel.1996.1998.Parliamentary Dem","Democracy","Parliamentary Dem",1996,3,1
"793","Israel",666,666,"Western Asia","Asia","Ehud Barak","Ehud Barak.Israel.1999.2000.Parliamentary Dem","Democracy","Parliamentary Dem",1999,2,1
"794","Israel",666,666,"Western Asia","Asia","Ariel Sharon","Ariel Sharon.Israel.2001.2005.Parliamentary Dem","Democracy","Parliamentary Dem",2001,5,1
"795","Israel",666,666,"Western Asia","Asia","Ehud Olmert","Ehud Olmert.Israel.2006.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2006,3,1
"796","Italy",325,325,"Southern Europe","Europe","Alcide de Gasperi","Alcide de Gasperi.Italy.1946.1952.Parliamentary Dem","Democracy","Parliamentary Dem",1946,7,1
"797","Italy",325,325,"Southern Europe","Europe","Giuseppe Pella","Giuseppe Pella.Italy.1953.1953.Parliamentary Dem","Democracy","Parliamentary Dem",1953,1,1
"798","Italy",325,325,"Southern Europe","Europe","Mario Scelba","Mario Scelba.Italy.1954.1954.Parliamentary Dem","Democracy","Parliamentary Dem",1954,1,1
"799","Italy",325,325,"Southern Europe","Europe","Antonio Segni","Antonio Segni.Italy.1955.1956.Parliamentary Dem","Democracy","Parliamentary Dem",1955,2,1
"800","Italy",325,325,"Southern Europe","Europe","Adone Zoli","Adone Zoli.Italy.1957.1957.Parliamentary Dem","Democracy","Parliamentary Dem",1957,1,1
"801","Italy",325,325,"Southern Europe","Europe","Amintore Fanfani","Amintore Fanfani.Italy.1958.1958.Parliamentary Dem","Democracy","Parliamentary Dem",1958,1,1
"802","Italy",325,325,"Southern Europe","Europe","Antonio Segni","Antonio Segni.Italy.1959.1959.Parliamentary Dem","Democracy","Parliamentary Dem",1959,1,1
"803","Italy",325,325,"Southern Europe","Europe","Amintore Fanfani","Amintore Fanfani.Italy.1960.1962.Parliamentary Dem","Democracy","Parliamentary Dem",1960,3,1
"804","Italy",325,325,"Southern Europe","Europe","Aldo Moro","Aldo Moro.Italy.1963.1967.Parliamentary Dem","Democracy","Parliamentary Dem",1963,5,1
"805","Italy",325,325,"Southern Europe","Europe","Mariano Rumor","Mariano Rumor.Italy.1968.1969.Parliamentary Dem","Democracy","Parliamentary Dem",1968,2,1
"806","Italy",325,325,"Southern Europe","Europe","Emilio Colombo","Emilio Colombo.Italy.1970.1971.Parliamentary Dem","Democracy","Parliamentary Dem",1970,2,1
"807","Italy",325,325,"Southern Europe","Europe","Giulio Andreotti","Giulio Andreotti.Italy.1972.1972.Parliamentary Dem","Democracy","Parliamentary Dem",1972,1,1
"808","Italy",325,325,"Southern Europe","Europe","Mariano Rumor","Mariano Rumor.Italy.1973.1973.Parliamentary Dem","Democracy","Parliamentary Dem",1973,1,1
"809","Italy",325,325,"Southern Europe","Europe","Aldo Moro","Aldo Moro.Italy.1974.1975.Parliamentary Dem","Democracy","Parliamentary Dem",1974,2,1
"810","Italy",325,325,"Southern Europe","Europe","Giulio Andreotti","Giulio Andreotti.Italy.1976.1978.Parliamentary Dem","Democracy","Parliamentary Dem",1976,3,1
"811","Italy",325,325,"Southern Europe","Europe","Francisco Cossiga","Francisco Cossiga.Italy.1979.1979.Parliamentary Dem","Democracy","Parliamentary Dem",1979,1,1
"812","Italy",325,325,"Southern Europe","Europe","Arnaldo Forlani","Arnaldo Forlani.Italy.1980.1980.Parliamentary Dem","Democracy","Parliamentary Dem",1980,1,1
"813","Italy",325,325,"Southern Europe","Europe","Giovanni Spadolini","Giovanni Spadolini.Italy.1981.1981.Parliamentary Dem","Democracy","Parliamentary Dem",1981,1,1
"814","Italy",325,325,"Southern Europe","Europe","Amintore Fanfani","Amintore Fanfani.Italy.1982.1982.Parliamentary Dem","Democracy","Parliamentary Dem",1982,1,1
"815","Italy",325,325,"Southern Europe","Europe","Benedetto Craxi","Benedetto Craxi.Italy.1983.1986.Parliamentary Dem","Democracy","Parliamentary Dem",1983,4,1
"816","Italy",325,325,"Southern Europe","Europe","Giovanni Goria","Giovanni Goria.Italy.1987.1987.Parliamentary Dem","Democracy","Parliamentary Dem",1987,1,1
"817","Italy",325,325,"Southern Europe","Europe","Ciriaco de Mita","Ciriaco de Mita.Italy.1988.1988.Parliamentary Dem","Democracy","Parliamentary Dem",1988,1,1
"818","Italy",325,325,"Southern Europe","Europe","Giulio Andreotti","Giulio Andreotti.Italy.1989.1991.Parliamentary Dem","Democracy","Parliamentary Dem",1989,3,1
"819","Italy",325,325,"Southern Europe","Europe","Giuliano Amato","Giuliano Amato.Italy.1992.1992.Parliamentary Dem","Democracy","Parliamentary Dem",1992,1,1
"820","Italy",325,325,"Southern Europe","Europe","Carlo Azeglio Ciampi","Carlo Azeglio Ciampi.Italy.1993.1993.Parliamentary Dem","Democracy","Parliamentary Dem",1993,1,1
"821","Italy",325,325,"Southern Europe","Europe","Silvio Berlusconi","Silvio Berlusconi.Italy.1994.1994.Parliamentary Dem","Democracy","Parliamentary Dem",1994,1,1
"822","Italy",325,325,"Southern Europe","Europe","Lamberto Dini","Lamberto Dini.Italy.1995.1995.Parliamentary Dem","Democracy","Parliamentary Dem",1995,1,1
"823","Italy",325,325,"Southern Europe","Europe","Romano Prodi","Romano Prodi.Italy.1996.1997.Parliamentary Dem","Democracy","Parliamentary Dem",1996,2,1
"824","Italy",325,325,"Southern Europe","Europe","Massimo D'Alema","Massimo D'Alema.Italy.1998.1999.Parliamentary Dem","Democracy","Parliamentary Dem",1998,2,1
"825","Italy",325,325,"Southern Europe","Europe","Guiliano Amato","Guiliano Amato.Italy.2000.2000.Parliamentary Dem","Democracy","Parliamentary Dem",2000,1,1
"826","Italy",325,325,"Southern Europe","Europe","Silvio Berlusconi","Silvio Berlusconi.Italy.2001.2005.Parliamentary Dem","Democracy","Parliamentary Dem",2001,5,1
"827","Italy",325,325,"Southern Europe","Europe","Romano Prodi","Romano Prodi.Italy.2006.2007.Parliamentary Dem","Democracy","Parliamentary Dem",2006,2,1
"828","Italy",325,325,"Southern Europe","Europe","Silvio Berlusconi","Silvio Berlusconi.Italy.2008.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2008,1,0
"829","Jamaica",51,51,"Caribbean","Americas","William Bustamante","William Bustamante.Jamaica.1962.1966.Parliamentary Dem","Democracy","Parliamentary Dem",1962,5,1
"830","Jamaica",51,51,"Caribbean","Americas","Hugh Lawson Shearer","Hugh Lawson Shearer.Jamaica.1967.1971.Parliamentary Dem","Democracy","Parliamentary Dem",1967,5,1
"831","Jamaica",51,51,"Caribbean","Americas","Michael Manley","Michael Manley.Jamaica.1972.1979.Parliamentary Dem","Democracy","Parliamentary Dem",1972,8,1
"832","Jamaica",51,51,"Caribbean","Americas","Edward Seaga","Edward Seaga.Jamaica.1980.1988.Parliamentary Dem","Democracy","Parliamentary Dem",1980,9,1
"833","Jamaica",51,51,"Caribbean","Americas","Michael Manley","Michael Manley.Jamaica.1989.1991.Parliamentary Dem","Democracy","Parliamentary Dem",1989,3,1
"834","Jamaica",51,51,"Caribbean","Americas","Percival Patterson","Percival Patterson.Jamaica.1992.2005.Parliamentary Dem","Democracy","Parliamentary Dem",1992,14,1
"835","Jamaica",51,51,"Caribbean","Americas","Portia Simpson Miller","Portia Simpson Miller.Jamaica.2006.2006.Parliamentary Dem","Democracy","Parliamentary Dem",2006,1,1
"836","Jamaica",51,51,"Caribbean","Americas","Orette Bruce Golding","Orette Bruce Golding.Jamaica.2007.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2007,2,0
"837","Japan",740,740,"Eastern Asia","Asia","Shigeru Yoshida","Shigeru Yoshida.Japan.1946.1946.Parliamentary Dem","Democracy","Parliamentary Dem",1946,1,0
"838","Japan",740,740,"Eastern Asia","Asia","Tetsu Katayama","Tetsu Katayama.Japan.1947.1947.Parliamentary Dem","Democracy","Parliamentary Dem",1947,1,1
"839","Japan",740,740,"Eastern Asia","Asia","Shigeru Yoshida","Shigeru Yoshida.Japan.1948.1953.Parliamentary Dem","Democracy","Parliamentary Dem",1948,6,1
"840","Japan",740,740,"Eastern Asia","Asia","Ichiro Hatoyama","Ichiro Hatoyama.Japan.1954.1955.Parliamentary Dem","Democracy","Parliamentary Dem",1954,2,1
"841","Japan",740,740,"Eastern Asia","Asia","Tanzan Ishibashi","Tanzan Ishibashi.Japan.1956.1956.Parliamentary Dem","Democracy","Parliamentary Dem",1956,1,1
"842","Japan",740,740,"Eastern Asia","Asia","Nobusuke Kishi","Nobusuke Kishi.Japan.1957.1959.Parliamentary Dem","Democracy","Parliamentary Dem",1957,3,1
"843","Japan",740,740,"Eastern Asia","Asia","Hayato Ikeda","Hayato Ikeda.Japan.1960.1963.Parliamentary Dem","Democracy","Parliamentary Dem",1960,4,1
"844","Japan",740,740,"Eastern Asia","Asia","Eisaku Sato","Eisaku Sato.Japan.1964.1971.Parliamentary Dem","Democracy","Parliamentary Dem",1964,8,1
"845","Japan",740,740,"Eastern Asia","Asia","Kakuei Tanaka","Kakuei Tanaka.Japan.1972.1973.Parliamentary Dem","Democracy","Parliamentary Dem",1972,2,1
"846","Japan",740,740,"Eastern Asia","Asia","Takeo Miki","Takeo Miki.Japan.1974.1975.Parliamentary Dem","Democracy","Parliamentary Dem",1974,2,1
"847","Japan",740,740,"Eastern Asia","Asia","Takeo Fukuda","Takeo Fukuda.Japan.1976.1977.Parliamentary Dem","Democracy","Parliamentary Dem",1976,2,1
"848","Japan",740,740,"Eastern Asia","Asia","Masayoshi Ohira","Masayoshi Ohira.Japan.1978.1979.Parliamentary Dem","Democracy","Parliamentary Dem",1978,2,0
"849","Japan",740,740,"Eastern Asia","Asia","Zenko Suzuki","Zenko Suzuki.Japan.1980.1981.Parliamentary Dem","Democracy","Parliamentary Dem",1980,2,1
"850","Japan",740,740,"Eastern Asia","Asia","Yasuhiro Nakasone","Yasuhiro Nakasone.Japan.1982.1986.Parliamentary Dem","Democracy","Parliamentary Dem",1982,5,1
"851","Japan",740,740,"Eastern Asia","Asia","Noboru Takeshita","Noboru Takeshita.Japan.1987.1988.Parliamentary Dem","Democracy","Parliamentary Dem",1987,2,1
"852","Japan",740,740,"Eastern Asia","Asia","Sosuke Uno","Sosuke Uno.Japan.1989.1990.Parliamentary Dem","Democracy","Parliamentary Dem",1989,2,1
"853","Japan",740,740,"Eastern Asia","Asia","Kiichi Miyazawa","Kiichi Miyazawa.Japan.1991.1992.Parliamentary Dem","Democracy","Parliamentary Dem",1991,2,1
"854","Japan",740,740,"Eastern Asia","Asia","Morihiro Hosokawa","Morihiro Hosokawa.Japan.1993.1993.Parliamentary Dem","Democracy","Parliamentary Dem",1993,1,1
"855","Japan",740,740,"Eastern Asia","Asia","Tomiichi Murayama","Tomiichi Murayama.Japan.1994.1995.Parliamentary Dem","Democracy","Parliamentary Dem",1994,2,1
"856","Japan",740,740,"Eastern Asia","Asia","Ryutaro Hashimoto","Ryutaro Hashimoto.Japan.1996.1997.Parliamentary Dem","Democracy","Parliamentary Dem",1996,2,1
"857","Japan",740,740,"Eastern Asia","Asia","Keizo Obuchi","Keizo Obuchi.Japan.1998.1999.Parliamentary Dem","Democracy","Parliamentary Dem",1998,2,0
"858","Japan",740,740,"Eastern Asia","Asia","Yoshiro Mori","Yoshiro Mori.Japan.2000.2000.Parliamentary Dem","Democracy","Parliamentary Dem",2000,1,1
"859","Japan",740,740,"Eastern Asia","Asia","Junichiro Koizumi","Junichiro Koizumi.Japan.2001.2005.Parliamentary Dem","Democracy","Parliamentary Dem",2001,5,1
"860","Japan",740,740,"Eastern Asia","Asia","Shinzo Abe","Shinzo Abe.Japan.2006.2006.Parliamentary Dem","Democracy","Parliamentary Dem",2006,1,1
"861","Japan",740,740,"Eastern Asia","Asia","Yasuo Fukuda","Yasuo Fukuda.Japan.2007.2007.Parliamentary Dem","Democracy","Parliamentary Dem",2007,1,1
"862","Japan",740,740,"Eastern Asia","Asia","Taro Aso","Taro Aso.Japan.2008.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2008,1,0
"863","Jordan",663,663,"Western Asia","Asia","Abdullah bin Hussein","Abdullah bin Hussein.Jordan.1946.1950.Monarchy","Non-democracy","Monarchy",1946,5,0
"864","Jordan",663,663,"Western Asia","Asia","Talal bin Abdullah","Talal bin Abdullah.Jordan.1951.1951.Monarchy","Non-democracy","Monarchy",1951,1,1
"865","Jordan",663,663,"Western Asia","Asia","Hussein bin Talal","Hussein bin Talal.Jordan.1952.1998.Monarchy","Non-democracy","Monarchy",1952,47,0
"866","Jordan",663,663,"Western Asia","Asia","Abdallah ibn al-Hussein al-Hashimi","Abdallah ibn al-Hussein al-Hashimi.Jordan.1999.2008.Monarchy","Non-democracy","Monarchy",1999,10,0
"867","Kazakhstan",705,705,"Central Asia","Asia","Nursultan Nazarbayev","Nursultan Nazarbayev.Kazakhstan.1991.2008.Civilian Dict","Non-democracy","Civilian Dict",1991,18,0
"868","Kenya",501,501,"Eastern Africa","Africa","Jomo Kenyatta","Jomo Kenyatta.Kenya.1963.1977.Civilian Dict","Non-democracy","Civilian Dict",1963,15,0
"869","Kenya",501,501,"Eastern Africa","Africa","Daniel Arap Moi","Daniel Arap Moi.Kenya.1978.1997.Civilian Dict","Non-democracy","Civilian Dict",1978,20,1
"870","Kenya",501,501,"Eastern Africa","Africa","Daniel Arap Moi","Daniel Arap Moi.Kenya.1998.2001.Presidential Dem","Democracy","Presidential Dem",1998,4,1
"871","Kenya",501,501,"Eastern Africa","Africa","Mwai Kibaki","Mwai Kibaki.Kenya.2002.2008.Presidential Dem","Democracy","Presidential Dem",2002,7,0
"872","Kiribati",946,946,"Micronesia","Oceania","Ieremia Tabai","Ieremia Tabai.Kiribati.1979.1981.Parliamentary Dem","Democracy","Parliamentary Dem",1979,3,1
"873","Kiribati",946,946,"Micronesia","Oceania","Rota Onorio","Rota Onorio.Kiribati.1982.1982.Parliamentary Dem","Democracy","Parliamentary Dem",1982,1,1
"874","Kiribati",946,946,"Micronesia","Oceania","Ieremia Tabai","Ieremia Tabai.Kiribati.1983.1990.Parliamentary Dem","Democracy","Parliamentary Dem",1983,8,1
"875","Kiribati",946,946,"Micronesia","Oceania","Teatao Teannaki","Teatao Teannaki.Kiribati.1991.1993.Parliamentary Dem","Democracy","Parliamentary Dem",1991,3,1
"876","Kiribati",946,946,"Micronesia","Oceania","Teburoro Tito","Teburoro Tito.Kiribati.1994.2002.Parliamentary Dem","Democracy","Parliamentary Dem",1994,9,1
"877","Kiribati",946,946,"Micronesia","Oceania","Anote Tong","Anote Tong.Kiribati.2003.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2003,6,0
"878","Kuwait",690,690,"Western Asia","Asia","Sheikh Abdullah Al Salim Al Sabah","Sheikh Abdullah Al Salim Al Sabah.Kuwait.1961.1964.Monarchy","Non-democracy","Monarchy",1961,4,0
"879","Kuwait",690,690,"Western Asia","Asia","Sheikh Sabah Al Salim Al Sabah","Sheikh Sabah Al Salim Al Sabah.Kuwait.1965.1976.Monarchy","Non-democracy","Monarchy",1965,12,0
"880","Kuwait",690,690,"Western Asia","Asia","Sheikh Jabir Al Ahmad Al Jabir Al Sabah","Sheikh Jabir Al Ahmad Al Jabir Al Sabah.Kuwait.1977.2005.Monarchy","Non-democracy","Monarchy",1977,29,1
"881","Kuwait",690,690,"Western Asia","Asia","Sheikh Sabah Al Ahmad Al Jabir Al Sabah","Sheikh Sabah Al Ahmad Al Jabir Al Sabah.Kuwait.2006.2008.Monarchy","Non-democracy","Monarchy",2006,3,0
"882","Kyrgyzstan",703,703,"Central Asia","Asia","Askar Akayev","Askar Akayev.Kyrgyzstan.1991.2004.Civilian Dict","Non-democracy","Civilian Dict",1991,14,1
"883","Kyrgyzstan",703,703,"Central Asia","Asia","Feliks Kulov","Feliks Kulov.Kyrgyzstan.2005.2006.Mixed Dem","Democracy","Mixed Dem",2005,2,1
"884","Kyrgyzstan",703,703,"Central Asia","Asia","Igor Chudinov","Igor Chudinov.Kyrgyzstan.2007.2008.Mixed Dem","Democracy","Mixed Dem",2007,2,0
"885","Laos",812,812,"South-Eastern Asia","Asia","Prince Souvanna Phouma","Prince Souvanna Phouma.Laos.1953.1953.Parliamentary Dem","Democracy","Parliamentary Dem",1953,1,1
"886","Laos",812,812,"South-Eastern Asia","Asia","Katay Don Sasorith","Katay Don Sasorith.Laos.1954.1955.Parliamentary Dem","Democracy","Parliamentary Dem",1954,2,1
"887","Laos",812,812,"South-Eastern Asia","Asia","Prince Souvanna Phouma","Prince Souvanna Phouma.Laos.1956.1957.Parliamentary Dem","Democracy","Parliamentary Dem",1956,2,1
"888","Laos",812,812,"South-Eastern Asia","Asia","Phoui Sananikone","Phoui Sananikone.Laos.1958.1958.Parliamentary Dem","Democracy","Parliamentary Dem",1958,1,1
"889","Laos",812,812,"South-Eastern Asia","Asia","Phoumi Nosavan","Phoumi Nosavan.Laos.1959.1961.Military Dict","Non-democracy","Military Dict",1959,3,1
"890","Laos",812,812,"South-Eastern Asia","Asia","Prince Souvanna Phouma","Prince Souvanna Phouma.Laos.1962.1974.Civilian Dict","Non-democracy","Civilian Dict",1962,13,1
"891","Laos",812,812,"South-Eastern Asia","Asia","Kaysone Phomvihan","Kaysone Phomvihan.Laos.1975.1991.Civilian Dict","Non-democracy","Civilian Dict",1975,17,0
"892","Laos",812,812,"South-Eastern Asia","Asia","Khamtay Siphandon","Khamtay Siphandon.Laos.1992.2005.Military Dict","Non-democracy","Military Dict",1992,14,1
"893","Laos",812,812,"South-Eastern Asia","Asia","Choummaly Sayasone","Choummaly Sayasone.Laos.2006.2008.Military Dict","Non-democracy","Military Dict",2006,3,0
"894","Latvia",367,367,"Northern Europe","Europe","Ivars Godmanis","Ivars Godmanis.Latvia.1991.1992.Parliamentary Dem","Democracy","Parliamentary Dem",1991,2,1
"895","Latvia",367,367,"Northern Europe","Europe","Valdis Birkavs","Valdis Birkavs.Latvia.1993.1993.Parliamentary Dem","Democracy","Parliamentary Dem",1993,1,1
"896","Latvia",367,367,"Northern Europe","Europe","Maris Gailis","Maris Gailis.Latvia.1994.1994.Parliamentary Dem","Democracy","Parliamentary Dem",1994,1,1
"897","Latvia",367,367,"Northern Europe","Europe","Andris Skele","Andris Skele.Latvia.1995.1996.Parliamentary Dem","Democracy","Parliamentary Dem",1995,2,1
"898","Latvia",367,367,"Northern Europe","Europe","Guntars Krasts","Guntars Krasts.Latvia.1997.1997.Parliamentary Dem","Democracy","Parliamentary Dem",1997,1,1
"899","Latvia",367,367,"Northern Europe","Europe","Vilis Kristopans","Vilis Kristopans.Latvia.1998.1998.Parliamentary Dem","Democracy","Parliamentary Dem",1998,1,1
"900","Latvia",367,367,"Northern Europe","Europe","Andris Skele","Andris Skele.Latvia.1999.1999.Parliamentary Dem","Democracy","Parliamentary Dem",1999,1,1
"901","Latvia",367,367,"Northern Europe","Europe","Andris Berzins","Andris Berzins.Latvia.2000.2001.Parliamentary Dem","Democracy","Parliamentary Dem",2000,2,1
"902","Latvia",367,367,"Northern Europe","Europe","Einars Repse","Einars Repse.Latvia.2002.2003.Parliamentary Dem","Democracy","Parliamentary Dem",2002,2,1
"903","Latvia",367,367,"Northern Europe","Europe","Aigars Kalvitis","Aigars Kalvitis.Latvia.2004.2006.Parliamentary Dem","Democracy","Parliamentary Dem",2004,3,1
"904","Latvia",367,367,"Northern Europe","Europe","Ivars Godmanis","Ivars Godmanis.Latvia.2007.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2007,2,0
"905","Lebanon",660,660,"Western Asia","Asia","Riyad es-Solh","Riyad es-Solh.Lebanon.1946.1950.Parliamentary Dem","Democracy","Parliamentary Dem",1946,5,1
"906","Lebanon",660,660,"Western Asia","Asia","Abdullah Aref al-Yafi","Abdullah Aref al-Yafi.Lebanon.1951.1951.Parliamentary Dem","Democracy","Parliamentary Dem",1951,1,1
"907","Lebanon",660,660,"Western Asia","Asia","Amir Khalid Chehab","Amir Khalid Chehab.Lebanon.1952.1952.Parliamentary Dem","Democracy","Parliamentary Dem",1952,1,1
"908","Lebanon",660,660,"Western Asia","Asia","Abdullah Aref al-Yafi","Abdullah Aref al-Yafi.Lebanon.1953.1953.Parliamentary Dem","Democracy","Parliamentary Dem",1953,1,1
"909","Lebanon",660,660,"Western Asia","Asia","Abd' Rashin Sami es-Solh","Abd' Rashin Sami es-Solh.Lebanon.1954.1954.Parliamentary Dem","Democracy","Parliamentary Dem",1954,1,1
"910","Lebanon",660,660,"Western Asia","Asia","Rashid Karame","Rashid Karame.Lebanon.1955.1955.Parliamentary Dem","Democracy","Parliamentary Dem",1955,1,1
"911","Lebanon",660,660,"Western Asia","Asia","Abd' Rashin Sami es-Solh","Abd' Rashin Sami es-Solh.Lebanon.1956.1957.Parliamentary Dem","Democracy","Parliamentary Dem",1956,2,1
"912","Lebanon",660,660,"Western Asia","Asia","Rashid Karame","Rashid Karame.Lebanon.1958.1959.Parliamentary Dem","Democracy","Parliamentary Dem",1958,2,1
"913","Lebanon",660,660,"Western Asia","Asia","Saeb Sallam","Saeb Sallam.Lebanon.1960.1960.Parliamentary Dem","Democracy","Parliamentary Dem",1960,1,1
"914","Lebanon",660,660,"Western Asia","Asia","Rashid Karame","Rashid Karame.Lebanon.1961.1963.Parliamentary Dem","Democracy","Parliamentary Dem",1961,3,1
"915","Lebanon",660,660,"Western Asia","Asia","Hussein al-Oweini","Hussein al-Oweini.Lebanon.1964.1964.Parliamentary Dem","Democracy","Parliamentary Dem",1964,1,1
"916","Lebanon",660,660,"Western Asia","Asia","Rashid Karame","Rashid Karame.Lebanon.1965.1967.Parliamentary Dem","Democracy","Parliamentary Dem",1965,3,1
"917","Lebanon",660,660,"Western Asia","Asia","Abdullah Aref al-Yafi","Abdullah Aref al-Yafi.Lebanon.1968.1968.Parliamentary Dem","Democracy","Parliamentary Dem",1968,1,1
"918","Lebanon",660,660,"Western Asia","Asia","Rashid Karame","Rashid Karame.Lebanon.1969.1969.Parliamentary Dem","Democracy","Parliamentary Dem",1969,1,1
"919","Lebanon",660,660,"Western Asia","Asia","Saeb Sallam","Saeb Sallam.Lebanon.1970.1972.Parliamentary Dem","Democracy","Parliamentary Dem",1970,3,1
"920","Lebanon",660,660,"Western Asia","Asia","Takieddin es-Solh","Takieddin es-Solh.Lebanon.1973.1973.Parliamentary Dem","Democracy","Parliamentary Dem",1973,1,1
"921","Lebanon",660,660,"Western Asia","Asia","Rashid es-Solh","Rashid es-Solh.Lebanon.1974.1974.Parliamentary Dem","Democracy","Parliamentary Dem",1974,1,1
"922","Lebanon",660,660,"Western Asia","Asia","Soleiman Kabalan Franjieh","Soleiman Kabalan Franjieh.Lebanon.1975.1975.Civilian Dict","Non-democracy","Civilian Dict",1975,1,1
"923","Lebanon",660,660,"Western Asia","Asia","Elias Sarkis","Elias Sarkis.Lebanon.1976.1981.Civilian Dict","Non-democracy","Civilian Dict",1976,6,1
"924","Lebanon",660,660,"Western Asia","Asia","Amine Pierre Gemayel","Amine Pierre Gemayel.Lebanon.1982.1987.Civilian Dict","Non-democracy","Civilian Dict",1982,6,1
"925","Lebanon",660,660,"Western Asia","Asia","Michel Aoun","Michel Aoun.Lebanon.1988.1988.Military Dict","Non-democracy","Military Dict",1988,1,1
"926","Lebanon",660,660,"Western Asia","Asia","Elias Khalil Haraoui","Elias Khalil Haraoui.Lebanon.1989.1997.Civilian Dict","Non-democracy","Civilian Dict",1989,9,1
"927","Lebanon",660,660,"Western Asia","Asia","Emile Geamil Lahoud","Emile Geamil Lahoud.Lebanon.1998.2006.Military Dict","Non-democracy","Military Dict",1998,9,1
"928","Lebanon",660,660,"Western Asia","Asia","Fouad Siniora","Fouad Siniora.Lebanon.2007.2007.Civilian Dict","Non-democracy","Civilian Dict",2007,1,1
"929","Lebanon",660,660,"Western Asia","Asia","Michel Suleiman","Michel Suleiman.Lebanon.2008.2008.Military Dict","Non-democracy","Military Dict",2008,1,0
"930","Lesotho",570,570,"Southern Africa","Africa","Leabua Jonathan","Leabua Jonathan.Lesotho.1966.1985.Civilian Dict","Non-democracy","Civilian Dict",1966,20,1
"931","Lesotho",570,570,"Southern Africa","Africa","Justin Metsino Lekhanya","Justin Metsino Lekhanya.Lesotho.1986.1990.Military Dict","Non-democracy","Military Dict",1986,5,1
"932","Lesotho",570,570,"Southern Africa","Africa","Elias Phisoana Ramaema","Elias Phisoana Ramaema.Lesotho.1991.1992.Military Dict","Non-democracy","Military Dict",1991,2,1
"933","Lesotho",570,570,"Southern Africa","Africa","Ntsu Mokhehle","Ntsu Mokhehle.Lesotho.1993.1997.Civilian Dict","Non-democracy","Civilian Dict",1993,5,1
"934","Lesotho",570,570,"Southern Africa","Africa","Pakalitha Mosisili","Pakalitha Mosisili.Lesotho.1998.2008.Civilian Dict","Non-democracy","Civilian Dict",1998,11,0
"935","Liberia",450,450,"Western Africa","Africa","William Tubman","William Tubman.Liberia.1946.1970.Civilian Dict","Non-democracy","Civilian Dict",1946,25,0
"936","Liberia",450,450,"Western Africa","Africa","William Tolbert, Jr.","William Tolbert, Jr..Liberia.1971.1979.Civilian Dict","Non-democracy","Civilian Dict",1971,9,0
"937","Liberia",450,450,"Western Africa","Africa","Samuel Doe","Samuel Doe.Liberia.1980.1989.Military Dict","Non-democracy","Military Dict",1980,10,0
"938","Liberia",450,450,"Western Africa","Africa","Amos Sawyer","Amos Sawyer.Liberia.1990.1992.Civilian Dict","Non-democracy","Civilian Dict",1990,3,1
"939","Liberia",450,450,"Western Africa","Africa","Philip Banks","Philip Banks.Liberia.1993.1993.Civilian Dict","Non-democracy","Civilian Dict",1993,1,1
"940","Liberia",450,450,"Western Africa","Africa","David Kpomakpor","David Kpomakpor.Liberia.1994.1994.Civilian Dict","Non-democracy","Civilian Dict",1994,1,1
"941","Liberia",450,450,"Western Africa","Africa","Wilton Sankawnlo","Wilton Sankawnlo.Liberia.1995.1995.Civilian Dict","Non-democracy","Civilian Dict",1995,1,1
"942","Liberia",450,450,"Western Africa","Africa","Ruth Perry","Ruth Perry.Liberia.1996.1996.Civilian Dict","Non-democracy","Civilian Dict",1996,1,1
"943","Liberia",450,450,"Western Africa","Africa","Charles Taylor","Charles Taylor.Liberia.1997.2002.Civilian Dict","Non-democracy","Civilian Dict",1997,6,1
"944","Liberia",450,450,"Western Africa","Africa","Charles Gyude Bryant","Charles Gyude Bryant.Liberia.2003.2005.Civilian Dict","Non-democracy","Civilian Dict",2003,3,1
"945","Liberia",450,450,"Western Africa","Africa","Ellen Johnson-Sirleaf","Ellen Johnson-Sirleaf.Liberia.2006.2008.Presidential Dem","Democracy","Presidential Dem",2006,3,0
"946","Libyan Arab Jamahiriya",620,620,"Northern Africa","Africa","Idris I","Idris I.Libyan Arab Jamahiriya.1951.1968.Monarchy","Non-democracy","Monarchy",1951,18,1
"947","Libyan Arab Jamahiriya",620,620,"Northern Africa","Africa","Muammar al-Qaddafi","Muammar al-Qaddafi.Libyan Arab Jamahiriya.1969.2008.Military Dict","Non-democracy","Military Dict",1969,40,0
"948","Liechtenstein",223,223,"Western Europe","Europe","Alexander Frick","Alexander Frick.Liechtenstein.1946.1961.Parliamentary Dem","Democracy","Parliamentary Dem",1946,16,1
"949","Liechtenstein",223,223,"Western Europe","Europe","Gerard Batliner","Gerard Batliner.Liechtenstein.1962.1969.Parliamentary Dem","Democracy","Parliamentary Dem",1962,8,1
"950","Liechtenstein",223,223,"Western Europe","Europe","Alfred Hilbe","Alfred Hilbe.Liechtenstein.1970.1973.Parliamentary Dem","Democracy","Parliamentary Dem",1970,4,1
"951","Liechtenstein",223,223,"Western Europe","Europe","Walter Kieber","Walter Kieber.Liechtenstein.1974.1977.Parliamentary Dem","Democracy","Parliamentary Dem",1974,4,1
"952","Liechtenstein",223,223,"Western Europe","Europe","Hans Brunhart","Hans Brunhart.Liechtenstein.1978.1992.Parliamentary Dem","Democracy","Parliamentary Dem",1978,15,1
"953","Liechtenstein",223,223,"Western Europe","Europe","Mario Frick","Mario Frick.Liechtenstein.1993.2000.Parliamentary Dem","Democracy","Parliamentary Dem",1993,8,1
"954","Liechtenstein",223,223,"Western Europe","Europe","Otmar Hasler","Otmar Hasler.Liechtenstein.2001.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2001,8,0
"955","Lithuania",368,368,"Northern Europe","Europe","Gediminas Vagnorius","Gediminas Vagnorius.Lithuania.1991.1991.Mixed Dem","Democracy","Mixed Dem",1991,1,1
"956","Lithuania",368,368,"Northern Europe","Europe","Bronislovas Lubys","Bronislovas Lubys.Lithuania.1992.1992.Mixed Dem","Democracy","Mixed Dem",1992,1,1
"957","Lithuania",368,368,"Northern Europe","Europe","Adolfas Slezevicius","Adolfas Slezevicius.Lithuania.1993.1995.Mixed Dem","Democracy","Mixed Dem",1993,3,1
"958","Lithuania",368,368,"Northern Europe","Europe","Gediminas Vagnorius","Gediminas Vagnorius.Lithuania.1996.1998.Mixed Dem","Democracy","Mixed Dem",1996,3,1
"959","Lithuania",368,368,"Northern Europe","Europe","Andrius Kubilius","Andrius Kubilius.Lithuania.1999.1999.Mixed Dem","Democracy","Mixed Dem",1999,1,1
"960","Lithuania",368,368,"Northern Europe","Europe","Rolandas Paksas","Rolandas Paksas.Lithuania.2000.2000.Mixed Dem","Democracy","Mixed Dem",2000,1,1
"961","Lithuania",368,368,"Northern Europe","Europe","Algirdas Brazauskas","Algirdas Brazauskas.Lithuania.2001.2005.Mixed Dem","Democracy","Mixed Dem",2001,5,1
"962","Lithuania",368,368,"Northern Europe","Europe","Gediminas Kirkilas","Gediminas Kirkilas.Lithuania.2006.2007.Mixed Dem","Democracy","Mixed Dem",2006,2,1
"963","Lithuania",368,368,"Northern Europe","Europe","Andrius Kubilius","Andrius Kubilius.Lithuania.2008.2008.Mixed Dem","Democracy","Mixed Dem",2008,1,0
"964","Luxembourg",212,212,"Western Europe","Europe","Pierre Dupong","Pierre Dupong.Luxembourg.1946.1952.Parliamentary Dem","Democracy","Parliamentary Dem",1946,7,0
"965","Luxembourg",212,212,"Western Europe","Europe","Joseph Bech","Joseph Bech.Luxembourg.1953.1957.Parliamentary Dem","Democracy","Parliamentary Dem",1953,5,1
"966","Luxembourg",212,212,"Western Europe","Europe","Pierre Frieden","Pierre Frieden.Luxembourg.1958.1958.Parliamentary Dem","Democracy","Parliamentary Dem",1958,1,1
"967","Luxembourg",212,212,"Western Europe","Europe","Pierre Werner","Pierre Werner.Luxembourg.1959.1973.Parliamentary Dem","Democracy","Parliamentary Dem",1959,15,1
"968","Luxembourg",212,212,"Western Europe","Europe","Gaston Thorn","Gaston Thorn.Luxembourg.1974.1978.Parliamentary Dem","Democracy","Parliamentary Dem",1974,5,1
"969","Luxembourg",212,212,"Western Europe","Europe","Pierre Werner","Pierre Werner.Luxembourg.1979.1983.Parliamentary Dem","Democracy","Parliamentary Dem",1979,5,1
"970","Luxembourg",212,212,"Western Europe","Europe","Jacques Santer","Jacques Santer.Luxembourg.1984.1994.Parliamentary Dem","Democracy","Parliamentary Dem",1984,11,1
"971","Luxembourg",212,212,"Western Europe","Europe","Jean-Claude Juncker","Jean-Claude Juncker.Luxembourg.1995.2008.Parliamentary Dem","Democracy","Parliamentary Dem",1995,14,0
"972","Macedonia",343,343,"Southern Europe","Europe","Nikola Kljusev","Nikola Kljusev.Macedonia.1991.1991.Mixed Dem","Democracy","Mixed Dem",1991,1,1
"973","Macedonia",343,343,"Southern Europe","Europe","Branko Crvenkovski","Branko Crvenkovski.Macedonia.1992.1997.Mixed Dem","Democracy","Mixed Dem",1992,6,1
"974","Macedonia",343,343,"Southern Europe","Europe","Ljupco Georgievski","Ljupco Georgievski.Macedonia.1998.2001.Mixed Dem","Democracy","Mixed Dem",1998,4,1
"975","Macedonia",343,343,"Southern Europe","Europe","Branko Crvenkovski","Branko Crvenkovski.Macedonia.2002.2003.Mixed Dem","Democracy","Mixed Dem",2002,2,1
"976","Macedonia",343,343,"Southern Europe","Europe","Vlado Buckovski","Vlado Buckovski.Macedonia.2004.2005.Mixed Dem","Democracy","Mixed Dem",2004,2,1
"977","Macedonia",343,343,"Southern Europe","Europe","Nikola Gruevski","Nikola Gruevski.Macedonia.2006.2008.Mixed Dem","Democracy","Mixed Dem",2006,3,0
"978","Madagascar",580,580,"Eastern Africa","Africa","Philibert Tsiranana","Philibert Tsiranana.Madagascar.1960.1971.Civilian Dict","Non-democracy","Civilian Dict",1960,12,1
"979","Madagascar",580,580,"Eastern Africa","Africa","Gabriel Ramanantsoa","Gabriel Ramanantsoa.Madagascar.1972.1974.Military Dict","Non-democracy","Military Dict",1972,3,1
"980","Madagascar",580,580,"Eastern Africa","Africa","Didier Ratsiraka","Didier Ratsiraka.Madagascar.1975.1992.Military Dict","Non-democracy","Military Dict",1975,18,1
"981","Madagascar",580,580,"Eastern Africa","Africa","Francisque Ravony","Francisque Ravony.Madagascar.1993.1994.Mixed Dem","Democracy","Mixed Dem",1993,2,1
"982","Madagascar",580,580,"Eastern Africa","Africa","Emmanuel Rakotovahiny","Emmanuel Rakotovahiny.Madagascar.1995.1995.Mixed Dem","Democracy","Mixed Dem",1995,1,1
"983","Madagascar",580,580,"Eastern Africa","Africa","Norbert Ratsirahonana","Norbert Ratsirahonana.Madagascar.1996.1996.Mixed Dem","Democracy","Mixed Dem",1996,1,1
"984","Madagascar",580,580,"Eastern Africa","Africa","Pascal Rakotomavo","Pascal Rakotomavo.Madagascar.1997.1997.Mixed Dem","Democracy","Mixed Dem",1997,1,1
"985","Madagascar",580,580,"Eastern Africa","Africa","Rene Tantely Andrianarivo","Rene Tantely Andrianarivo.Madagascar.1998.2001.Mixed Dem","Democracy","Mixed Dem",1998,4,1
"986","Madagascar",580,580,"Eastern Africa","Africa","Jacques Sylla","Jacques Sylla.Madagascar.2002.2006.Mixed Dem","Democracy","Mixed Dem",2002,5,1
"987","Madagascar",580,580,"Eastern Africa","Africa","Charles Rabemananjara","Charles Rabemananjara.Madagascar.2007.2008.Mixed Dem","Democracy","Mixed Dem",2007,2,0
"988","Malawi",553,553,"Eastern Africa","Africa","H. Kamuzu Banda","H. Kamuzu Banda.Malawi.1964.1993.Civilian Dict","Non-democracy","Civilian Dict",1964,30,1
"989","Malawi",553,553,"Eastern Africa","Africa","Bakili Muluzi","Bakili Muluzi.Malawi.1994.2003.Presidential Dem","Democracy","Presidential Dem",1994,10,1
"990","Malawi",553,553,"Eastern Africa","Africa","Bingu wa Mutharika","Bingu wa Mutharika.Malawi.2004.2008.Presidential Dem","Democracy","Presidential Dem",2004,5,0
"991","Malaysia",820,820,"South-Eastern Asia","Asia","Tunku Abdul Rahman Putra","Tunku Abdul Rahman Putra.Malaysia.1957.1969.Civilian Dict","Non-democracy","Civilian Dict",1957,13,1
"992","Malaysia",820,820,"South-Eastern Asia","Asia","Tun Abdul Razak bin Hussein","Tun Abdul Razak bin Hussein.Malaysia.1970.1975.Civilian Dict","Non-democracy","Civilian Dict",1970,6,0
"993","Malaysia",820,820,"South-Eastern Asia","Asia","Hussein bin Onn","Hussein bin Onn.Malaysia.1976.1980.Civilian Dict","Non-democracy","Civilian Dict",1976,5,1
"994","Malaysia",820,820,"South-Eastern Asia","Asia","Mahathir bin Mohamed","Mahathir bin Mohamed.Malaysia.1981.2002.Civilian Dict","Non-democracy","Civilian Dict",1981,22,1
"995","Malaysia",820,820,"South-Eastern Asia","Asia","Abdullah Ahmad Badawi","Abdullah Ahmad Badawi.Malaysia.2003.2008.Civilian Dict","Non-democracy","Civilian Dict",2003,6,0
"996","Maldives",781,781,"Southern Asia","Asia","Muhammad Farid Didi","Muhammad Farid Didi.Maldives.1965.1967.Monarchy","Non-democracy","Monarchy",1965,3,1
"997","Maldives",781,781,"Southern Asia","Asia","Ibrahim Nasir","Ibrahim Nasir.Maldives.1968.1977.Civilian Dict","Non-democracy","Civilian Dict",1968,10,1
"998","Maldives",781,781,"Southern Asia","Asia","Maumoon Abdul Gayoom","Maumoon Abdul Gayoom.Maldives.1978.2007.Civilian Dict","Non-democracy","Civilian Dict",1978,30,1
"999","Maldives",781,781,"Southern Asia","Asia","Mohamed Nasheed","Mohamed Nasheed.Maldives.2008.2008.Presidential Dem","Democracy","Presidential Dem",2008,1,0
"1000","Mali",432,432,"Western Africa","Africa","Modibo Keita","Modibo Keita.Mali.1960.1967.Civilian Dict","Non-democracy","Civilian Dict",1960,8,1
"1001","Mali",432,432,"Western Africa","Africa","Moussa Traore","Moussa Traore.Mali.1968.1990.Military Dict","Non-democracy","Military Dict",1968,23,1
"1002","Mali",432,432,"Western Africa","Africa","Amadou Toumani Toure","Amadou Toumani Toure.Mali.1991.1991.Military Dict","Non-democracy","Military Dict",1991,1,1
"1003","Mali",432,432,"Western Africa","Africa","Younoussi Toure","Younoussi Toure.Mali.1992.1992.Mixed Dem","Democracy","Mixed Dem",1992,1,1
"1004","Mali",432,432,"Western Africa","Africa","Abdoulaye Sekou Sow","Abdoulaye Sekou Sow.Mali.1993.1993.Mixed Dem","Democracy","Mixed Dem",1993,1,1
"1005","Mali",432,432,"Western Africa","Africa","Ibrahima Boubacar Keita","Ibrahima Boubacar Keita.Mali.1994.1999.Mixed Dem","Democracy","Mixed Dem",1994,6,1
"1006","Mali",432,432,"Western Africa","Africa","Mande Sidibe","Mande Sidibe.Mali.2000.2001.Mixed Dem","Democracy","Mixed Dem",2000,2,1
"1007","Mali",432,432,"Western Africa","Africa","Ahmed Mohamed Ag Hamani","Ahmed Mohamed Ag Hamani.Mali.2002.2003.Mixed Dem","Democracy","Mixed Dem",2002,2,1
"1008","Mali",432,432,"Western Africa","Africa","Ousmane Issoufi Maiga","Ousmane Issoufi Maiga.Mali.2004.2006.Mixed Dem","Democracy","Mixed Dem",2004,3,1
"1009","Mali",432,432,"Western Africa","Africa","Modibo Sidibe","Modibo Sidibe.Mali.2007.2008.Mixed Dem","Democracy","Mixed Dem",2007,2,0
"1010","Malta",338,338,"Southern Europe","Europe","George Borg Olivier","George Borg Olivier.Malta.1964.1970.Parliamentary Dem","Democracy","Parliamentary Dem",1964,7,1
"1011","Malta",338,338,"Southern Europe","Europe","Dominic Mintoff","Dominic Mintoff.Malta.1971.1983.Parliamentary Dem","Democracy","Parliamentary Dem",1971,13,1
"1012","Malta",338,338,"Southern Europe","Europe","Carmelo Mifsud-Bonnici","Carmelo Mifsud-Bonnici.Malta.1984.1986.Parliamentary Dem","Democracy","Parliamentary Dem",1984,3,1
"1013","Malta",338,338,"Southern Europe","Europe","Eddie Fenech Adami","Eddie Fenech Adami.Malta.1987.1995.Parliamentary Dem","Democracy","Parliamentary Dem",1987,9,1
"1014","Malta",338,338,"Southern Europe","Europe","Alfred Sant","Alfred Sant.Malta.1996.1997.Parliamentary Dem","Democracy","Parliamentary Dem",1996,2,1
"1015","Malta",338,338,"Southern Europe","Europe","Eddie Fenech Adami","Eddie Fenech Adami.Malta.1998.2003.Parliamentary Dem","Democracy","Parliamentary Dem",1998,6,1
"1016","Malta",338,338,"Southern Europe","Europe","Lawrence Gonzi","Lawrence Gonzi.Malta.2004.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2004,5,0
"1017","Marshall Islands",983,983,"Micronesia","Oceania","Amata Kabua","Amata Kabua.Marshall Islands.1990.1995.Parliamentary Dem","Democracy","Parliamentary Dem",1990,6,0
"1018","Marshall Islands",983,983,"Micronesia","Oceania","Kunio Lemari","Kunio Lemari.Marshall Islands.1996.1996.Parliamentary Dem","Democracy","Parliamentary Dem",1996,1,1
"1019","Marshall Islands",983,983,"Micronesia","Oceania","Imata Kabua","Imata Kabua.Marshall Islands.1997.1999.Parliamentary Dem","Democracy","Parliamentary Dem",1997,3,1
"1020","Marshall Islands",983,983,"Micronesia","Oceania","Kessai Note","Kessai Note.Marshall Islands.2000.2007.Parliamentary Dem","Democracy","Parliamentary Dem",2000,8,1
"1021","Marshall Islands",983,983,"Micronesia","Oceania","Litokwa Tomeing","Litokwa Tomeing.Marshall Islands.2008.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2008,1,0
"1022","Mauritania",435,435,"Western Africa","Africa","Moktar Ould Daddah","Moktar Ould Daddah.Mauritania.1960.1977.Civilian Dict","Non-democracy","Civilian Dict",1960,18,1
"1023","Mauritania",435,435,"Western Africa","Africa","Mustapha Ould Salek","Mustapha Ould Salek.Mauritania.1978.1978.Military Dict","Non-democracy","Military Dict",1978,1,1
"1024","Mauritania",435,435,"Western Africa","Africa","Mahmed Ould Louly","Mahmed Ould Louly.Mauritania.1979.1979.Military Dict","Non-democracy","Military Dict",1979,1,1
"1025","Mauritania",435,435,"Western Africa","Africa","Mohammed Khouna Ould Haydalla","Mohammed Khouna Ould Haydalla.Mauritania.1980.1983.Military Dict","Non-democracy","Military Dict",1980,4,1
"1026","Mauritania",435,435,"Western Africa","Africa","Moaouya Ould Sidi Ahmed Taya","Moaouya Ould Sidi Ahmed Taya.Mauritania.1984.2004.Military Dict","Non-democracy","Military Dict",1984,21,1
"1027","Mauritania",435,435,"Western Africa","Africa","Ely Ould Mohamed Vall","Ely Ould Mohamed Vall.Mauritania.2005.2006.Military Dict","Non-democracy","Military Dict",2005,2,1
"1028","Mauritania",435,435,"Western Africa","Africa","Zeine Ould Zeidane","Zeine Ould Zeidane.Mauritania.2007.2007.Mixed Dem","Democracy","Mixed Dem",2007,1,1
"1029","Mauritania",435,435,"Western Africa","Africa","Mohamed Ould Abdel Aziz","Mohamed Ould Abdel Aziz.Mauritania.2008.2008.Military Dict","Non-democracy","Military Dict",2008,1,0
"1030","Mauritius",590,590,"Eastern Africa","Africa","Seewoosagur Ramgoolam","Seewoosagur Ramgoolam.Mauritius.1968.1981.Parliamentary Dem","Democracy","Parliamentary Dem",1968,14,1
"1031","Mauritius",590,590,"Eastern Africa","Africa","Anerood Jugnauth","Anerood Jugnauth.Mauritius.1982.1994.Parliamentary Dem","Democracy","Parliamentary Dem",1982,13,1
"1032","Mauritius",590,590,"Eastern Africa","Africa","Navin Ramgoolam","Navin Ramgoolam.Mauritius.1995.1999.Parliamentary Dem","Democracy","Parliamentary Dem",1995,5,1
"1033","Mauritius",590,590,"Eastern Africa","Africa","Anerood Jugnauth","Anerood Jugnauth.Mauritius.2000.2002.Parliamentary Dem","Democracy","Parliamentary Dem",2000,3,1
"1034","Mauritius",590,590,"Eastern Africa","Africa","Paul Raymond Berenger","Paul Raymond Berenger.Mauritius.2003.2004.Parliamentary Dem","Democracy","Parliamentary Dem",2003,2,1
"1035","Mauritius",590,590,"Eastern Africa","Africa","Navin(chandra) Ramgoolam","Navin(chandra) Ramgoolam.Mauritius.2005.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2005,4,0
"1036","Mexico",70,70,"Central America","Americas","Miguel Aleman","Miguel Aleman.Mexico.1946.1951.Civilian Dict","Non-democracy","Civilian Dict",1946,6,1
"1037","Mexico",70,70,"Central America","Americas","Adolfo Ruiz Cortines","Adolfo Ruiz Cortines.Mexico.1952.1957.Military Dict","Non-democracy","Military Dict",1952,6,1
"1038","Mexico",70,70,"Central America","Americas","Adolfo Lopez Mateos","Adolfo Lopez Mateos.Mexico.1958.1963.Civilian Dict","Non-democracy","Civilian Dict",1958,6,1
"1039","Mexico",70,70,"Central America","Americas","Gustavo Diaz Ordaz","Gustavo Diaz Ordaz.Mexico.1964.1969.Civilian Dict","Non-democracy","Civilian Dict",1964,6,1
"1040","Mexico",70,70,"Central America","Americas","Luis Echeverria Alvarez","Luis Echeverria Alvarez.Mexico.1970.1975.Civilian Dict","Non-democracy","Civilian Dict",1970,6,1
"1041","Mexico",70,70,"Central America","Americas","Jose Lopez Portillo","Jose Lopez Portillo.Mexico.1976.1981.Civilian Dict","Non-democracy","Civilian Dict",1976,6,1
"1042","Mexico",70,70,"Central America","Americas","Miguel de la Madrid","Miguel de la Madrid.Mexico.1982.1987.Civilian Dict","Non-democracy","Civilian Dict",1982,6,1
"1043","Mexico",70,70,"Central America","Americas","Carlos Salinas de Gortari","Carlos Salinas de Gortari.Mexico.1988.1993.Civilian Dict","Non-democracy","Civilian Dict",1988,6,1
"1044","Mexico",70,70,"Central America","Americas","Ernesto Zedillo","Ernesto Zedillo.Mexico.1994.1999.Civilian Dict","Non-democracy","Civilian Dict",1994,6,1
"1045","Mexico",70,70,"Central America","Americas","Vicente Fox Quesada","Vicente Fox Quesada.Mexico.2000.2005.Presidential Dem","Democracy","Presidential Dem",2000,6,1
"1046","Mexico",70,70,"Central America","Americas","Felipe de Jesï¿½s Calderï¿½n Hinojosa","Felipe de Jesï¿½s Calderï¿½n Hinojosa.Mexico.2006.2008.Presidential Dem","Democracy","Presidential Dem",2006,3,0
"1047","Micronesia, Federated States of",987,987,"Micronesia","Oceania","Bailey Olter","Bailey Olter.Micronesia, Federated States of.1991.1995.Presidential Dem","Democracy","Presidential Dem",1991,5,1
"1048","Micronesia, Federated States of",987,987,"Micronesia","Oceania","Jacob Nena","Jacob Nena.Micronesia, Federated States of.1996.1998.Presidential Dem","Democracy","Presidential Dem",1996,3,1
"1049","Micronesia, Federated States of",987,987,"Micronesia","Oceania","Leo Falcam","Leo Falcam.Micronesia, Federated States of.1999.2002.Presidential Dem","Democracy","Presidential Dem",1999,4,1
"1050","Micronesia, Federated States of",987,987,"Micronesia","Oceania","Joseph J. Urusemal","Joseph J. Urusemal.Micronesia, Federated States of.2003.2006.Presidential Dem","Democracy","Presidential Dem",2003,4,1
"1051","Micronesia, Federated States of",987,987,"Micronesia","Oceania","Manny Mori","Manny Mori.Micronesia, Federated States of.2007.2008.Presidential Dem","Democracy","Presidential Dem",2007,2,0
"1052","Moldova",359,359,"Eastern Europe","Europe","Mirecea Snegur","Mirecea Snegur.Moldova.1991.1995.Parliamentary Dem","Democracy","Parliamentary Dem",1991,5,1
"1053","Moldova",359,359,"Eastern Europe","Europe","Andrei Sangheli","Andrei Sangheli.Moldova.1996.1996.Parliamentary Dem","Democracy","Parliamentary Dem",1996,1,1
"1054","Moldova",359,359,"Eastern Europe","Europe","Ion Ciubuc","Ion Ciubuc.Moldova.1997.1998.Mixed Dem","Democracy","Mixed Dem",1997,2,1
"1055","Moldova",359,359,"Eastern Europe","Europe","Dumitru Braghis","Dumitru Braghis.Moldova.1999.1999.Mixed Dem","Democracy","Mixed Dem",1999,1,1
"1056","Moldova",359,359,"Eastern Europe","Europe","Dumitru Braghis","Dumitru Braghis.Moldova.2000.2000.Parliamentary Dem","Democracy","Parliamentary Dem",2000,1,1
"1057","Moldova",359,359,"Eastern Europe","Europe","Vasile Tarlev","Vasile Tarlev.Moldova.2001.2007.Parliamentary Dem","Democracy","Parliamentary Dem",2001,7,1
"1058","Moldova",359,359,"Eastern Europe","Europe","Zinaida Greceanii","Zinaida Greceanii.Moldova.2008.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2008,1,0
"1059","Mongolia",712,712,"Eastern Asia","Asia","Yumjaagiyn Tsedenbal","Yumjaagiyn Tsedenbal.Mongolia.1946.1953.Civilian Dict","Non-democracy","Civilian Dict",1946,8,1
"1060","Mongolia",712,712,"Eastern Asia","Asia","Dashiyn Damba","Dashiyn Damba.Mongolia.1954.1957.Civilian Dict","Non-democracy","Civilian Dict",1954,4,1
"1061","Mongolia",712,712,"Eastern Asia","Asia","Yumjaagiyn Tsedenbal","Yumjaagiyn Tsedenbal.Mongolia.1958.1983.Civilian Dict","Non-democracy","Civilian Dict",1958,26,1
"1062","Mongolia",712,712,"Eastern Asia","Asia","Jambyn Batmounkh","Jambyn Batmounkh.Mongolia.1984.1989.Civilian Dict","Non-democracy","Civilian Dict",1984,6,1
"1063","Mongolia",712,712,"Eastern Asia","Asia","Punsalmaagiyn Ochirbat","Punsalmaagiyn Ochirbat.Mongolia.1990.1991.Parliamentary Dem","Democracy","Parliamentary Dem",1990,2,1
"1064","Mongolia",712,712,"Eastern Asia","Asia","Puntsagiyn Jasray","Puntsagiyn Jasray.Mongolia.1992.1995.Mixed Dem","Democracy","Mixed Dem",1992,4,1
"1065","Mongolia",712,712,"Eastern Asia","Asia","Mendsayhany Enkhsaikhan","Mendsayhany Enkhsaikhan.Mongolia.1996.1997.Mixed Dem","Democracy","Mixed Dem",1996,2,1
"1066","Mongolia",712,712,"Eastern Asia","Asia","Janlaviyn Narantsatsralt","Janlaviyn Narantsatsralt.Mongolia.1998.1998.Mixed Dem","Democracy","Mixed Dem",1998,1,1
"1067","Mongolia",712,712,"Eastern Asia","Asia","Rinchinnyamiyn Amarjargal","Rinchinnyamiyn Amarjargal.Mongolia.1999.1999.Mixed Dem","Democracy","Mixed Dem",1999,1,1
"1068","Mongolia",712,712,"Eastern Asia","Asia","Nambaryn Enkhbayar","Nambaryn Enkhbayar.Mongolia.2000.2003.Mixed Dem","Democracy","Mixed Dem",2000,4,1
"1069","Mongolia",712,712,"Eastern Asia","Asia","Tsakhiagiyn Elbegdorj","Tsakhiagiyn Elbegdorj.Mongolia.2004.2005.Mixed Dem","Democracy","Mixed Dem",2004,2,1
"1070","Mongolia",712,712,"Eastern Asia","Asia","Miyeegombo Enkhbold","Miyeegombo Enkhbold.Mongolia.2006.2006.Mixed Dem","Democracy","Mixed Dem",2006,1,1
"1071","Mongolia",712,712,"Eastern Asia","Asia","Sanj Bayar","Sanj Bayar.Mongolia.2007.2008.Mixed Dem","Democracy","Mixed Dem",2007,2,0
"1072","Montenegro",341,NA,"Southern Europe","Europe","Filip Vujanovic","Filip Vujanovic.Montenegro.2006.2008.Civilian Dict","Non-democracy","Civilian Dict",2006,3,0
"1073","Morocco",600,600,"Northern Africa","Africa","Muhammad V","Muhammad V.Morocco.1956.1960.Monarchy","Non-democracy","Monarchy",1956,5,0
"1074","Morocco",600,600,"Northern Africa","Africa","Hassan II","Hassan II.Morocco.1961.1998.Monarchy","Non-democracy","Monarchy",1961,38,0
"1075","Morocco",600,600,"Northern Africa","Africa","Muhammad VI","Muhammad VI.Morocco.1999.2008.Monarchy","Non-democracy","Monarchy",1999,10,0
"1076","Mozambique",541,541,"Eastern Africa","Africa","Samora Machel","Samora Machel.Mozambique.1975.1985.Civilian Dict","Non-democracy","Civilian Dict",1975,11,0
"1077","Mozambique",541,541,"Eastern Africa","Africa","Joaquim Chissano","Joaquim Chissano.Mozambique.1986.2004.Civilian Dict","Non-democracy","Civilian Dict",1986,19,1
"1078","Mozambique",541,541,"Eastern Africa","Africa","Armando Emï¿½lio Guebuza","Armando Emï¿½lio Guebuza.Mozambique.2005.2008.Civilian Dict","Non-democracy","Civilian Dict",2005,4,0
"1079","Myanmar",775,775,"South-Eastern Asia","Asia","U Nu","U Nu.Myanmar.1948.1955.Parliamentary Dem","Democracy","Parliamentary Dem",1948,8,1
"1080","Myanmar",775,775,"South-Eastern Asia","Asia","U Ba Swe","U Ba Swe.Myanmar.1956.1956.Parliamentary Dem","Democracy","Parliamentary Dem",1956,1,1
"1081","Myanmar",775,775,"South-Eastern Asia","Asia","U Nu","U Nu.Myanmar.1957.1957.Parliamentary Dem","Democracy","Parliamentary Dem",1957,1,1
"1082","Myanmar",775,775,"South-Eastern Asia","Asia","Ne Win","Ne Win.Myanmar.1958.1959.Military Dict","Non-democracy","Military Dict",1958,2,1
"1083","Myanmar",775,775,"South-Eastern Asia","Asia","U Nu","U Nu.Myanmar.1960.1961.Parliamentary Dem","Democracy","Parliamentary Dem",1960,2,1
"1084","Myanmar",775,775,"South-Eastern Asia","Asia","Ne Win","Ne Win.Myanmar.1962.1987.Military Dict","Non-democracy","Military Dict",1962,26,1
"1085","Myanmar",775,775,"South-Eastern Asia","Asia","Khin Nyunt","Khin Nyunt.Myanmar.1988.1991.Military Dict","Non-democracy","Military Dict",1988,4,1
"1086","Myanmar",775,775,"South-Eastern Asia","Asia","Than Shwe","Than Shwe.Myanmar.1992.2008.Military Dict","Non-democracy","Military Dict",1992,17,0
"1087","Namibia",565,565,"Southern Africa","Africa","Sam Nujoma","Sam Nujoma.Namibia.1990.2004.Civilian Dict","Non-democracy","Civilian Dict",1990,15,1
"1088","Namibia",565,565,"Southern Africa","Africa","Hifikepunye Lucas Pohamba","Hifikepunye Lucas Pohamba.Namibia.2005.2008.Civilian Dict","Non-democracy","Civilian Dict",2005,4,0
"1089","Nauru",970,970,"Micronesia","Oceania","Hammer DeRoburt","Hammer DeRoburt.Nauru.1968.1975.Parliamentary Dem","Democracy","Parliamentary Dem",1968,8,1
"1090","Nauru",970,970,"Micronesia","Oceania","Bernard Dowiyogo","Bernard Dowiyogo.Nauru.1976.1977.Parliamentary Dem","Democracy","Parliamentary Dem",1976,2,1
"1091","Nauru",970,970,"Micronesia","Oceania","Hammer DeRoburt","Hammer DeRoburt.Nauru.1978.1988.Parliamentary Dem","Democracy","Parliamentary Dem",1978,11,1
"1092","Nauru",970,970,"Micronesia","Oceania","Bernard Dowiyoyo","Bernard Dowiyoyo.Nauru.1989.1994.Parliamentary Dem","Democracy","Parliamentary Dem",1989,6,1
"1093","Nauru",970,970,"Micronesia","Oceania","Lagumot Harris","Lagumot Harris.Nauru.1995.1995.Parliamentary Dem","Democracy","Parliamentary Dem",1995,1,1
"1094","Nauru",970,970,"Micronesia","Oceania","Rueben Kun","Rueben Kun.Nauru.1996.1996.Parliamentary Dem","Democracy","Parliamentary Dem",1996,1,1
"1095","Nauru",970,970,"Micronesia","Oceania","Kinza Clodumar","Kinza Clodumar.Nauru.1997.1997.Parliamentary Dem","Democracy","Parliamentary Dem",1997,1,1
"1096","Nauru",970,970,"Micronesia","Oceania","Bernard Dowiyoyo","Bernard Dowiyoyo.Nauru.1998.1998.Parliamentary Dem","Democracy","Parliamentary Dem",1998,1,1
"1097","Nauru",970,970,"Micronesia","Oceania","Rene Harris","Rene Harris.Nauru.1999.1999.Parliamentary Dem","Democracy","Parliamentary Dem",1999,1,1
"1098","Nauru",970,970,"Micronesia","Oceania","Bernard Dowiyoyo","Bernard Dowiyoyo.Nauru.2000.2000.Parliamentary Dem","Democracy","Parliamentary Dem",2000,1,1
"1099","Nauru",970,970,"Micronesia","Oceania","Rene Harris","Rene Harris.Nauru.2001.2003.Parliamentary Dem","Democracy","Parliamentary Dem",2001,3,1
"1100","Nauru",970,970,"Micronesia","Oceania","Ludwig Scotty","Ludwig Scotty.Nauru.2004.2006.Parliamentary Dem","Democracy","Parliamentary Dem",2004,3,1
"1101","Nauru",970,970,"Micronesia","Oceania","Marcus Stephen","Marcus Stephen.Nauru.2007.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2007,2,0
"1102","Nepal",790,790,"Southern Asia","Asia","Padma Shumshere Rana","Padma Shumshere Rana.Nepal.1946.1947.Civilian Dict","Non-democracy","Civilian Dict",1946,2,1
"1103","Nepal",790,790,"Southern Asia","Asia","Mohan Shumshere Rana","Mohan Shumshere Rana.Nepal.1948.1950.Civilian Dict","Non-democracy","Civilian Dict",1948,3,1
"1104","Nepal",790,790,"Southern Asia","Asia","Tribhuwan Bir Bikram Shah Deva","Tribhuwan Bir Bikram Shah Deva.Nepal.1951.1954.Monarchy","Non-democracy","Monarchy",1951,4,0
"1105","Nepal",790,790,"Southern Asia","Asia","Mahendra Bir Bikram Shah Deva","Mahendra Bir Bikram Shah Deva.Nepal.1955.1971.Monarchy","Non-democracy","Monarchy",1955,17,0
"1106","Nepal",790,790,"Southern Asia","Asia","Birendra Bir Bikram Shah Deva","Birendra Bir Bikram Shah Deva.Nepal.1972.1989.Monarchy","Non-democracy","Monarchy",1972,18,1
"1107","Nepal",790,790,"Southern Asia","Asia","Birendra Bir Bikram Shah Deva","Birendra Bir Bikram Shah Deva.Nepal.1990.1990.Parliamentary Dem","Democracy","Parliamentary Dem",1990,1,1
"1108","Nepal",790,790,"Southern Asia","Asia","Girija Prasad Koirala","Girija Prasad Koirala.Nepal.1991.1993.Parliamentary Dem","Democracy","Parliamentary Dem",1991,3,1
"1109","Nepal",790,790,"Southern Asia","Asia","Man Mohan Adhikari","Man Mohan Adhikari.Nepal.1994.1994.Parliamentary Dem","Democracy","Parliamentary Dem",1994,1,1
"1110","Nepal",790,790,"Southern Asia","Asia","Sher Bahadur Deuba","Sher Bahadur Deuba.Nepal.1995.1996.Parliamentary Dem","Democracy","Parliamentary Dem",1995,2,1
"1111","Nepal",790,790,"Southern Asia","Asia","Surya Bahadur Thapa","Surya Bahadur Thapa.Nepal.1997.1997.Parliamentary Dem","Democracy","Parliamentary Dem",1997,1,1
"1112","Nepal",790,790,"Southern Asia","Asia","Girija Prasad Koirala","Girija Prasad Koirala.Nepal.1998.1998.Parliamentary Dem","Democracy","Parliamentary Dem",1998,1,1
"1113","Nepal",790,790,"Southern Asia","Asia","Krishna Prasad Bhattarai","Krishna Prasad Bhattarai.Nepal.1999.1999.Parliamentary Dem","Democracy","Parliamentary Dem",1999,1,1
"1114","Nepal",790,790,"Southern Asia","Asia","Girija Prasad Koirala","Girija Prasad Koirala.Nepal.2000.2000.Parliamentary Dem","Democracy","Parliamentary Dem",2000,1,1
"1115","Nepal",790,790,"Southern Asia","Asia","Sher Bahadur Deuba","Sher Bahadur Deuba.Nepal.2001.2001.Parliamentary Dem","Democracy","Parliamentary Dem",2001,1,1
"1116","Nepal",790,790,"Southern Asia","Asia","Gyanendra Bir Bikram Shah Deva","Gyanendra Bir Bikram Shah Deva.Nepal.2002.2007.Monarchy","Non-democracy","Monarchy",2002,6,1
"1117","Nepal",790,790,"Southern Asia","Asia","Pushpa Kamal Dahal","Pushpa Kamal Dahal.Nepal.2008.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2008,1,0
"1118","Netherlands",210,210,"Western Europe","Europe","Louis Beel","Louis Beel.Netherlands.1946.1947.Parliamentary Dem","Democracy","Parliamentary Dem",1946,2,1
"1119","Netherlands",210,210,"Western Europe","Europe","Willem Drees","Willem Drees.Netherlands.1948.1957.Parliamentary Dem","Democracy","Parliamentary Dem",1948,10,1
"1120","Netherlands",210,210,"Western Europe","Europe","Louis Beel","Louis Beel.Netherlands.1958.1958.Parliamentary Dem","Democracy","Parliamentary Dem",1958,1,1
"1121","Netherlands",210,210,"Western Europe","Europe","Jan de Quay","Jan de Quay.Netherlands.1959.1962.Parliamentary Dem","Democracy","Parliamentary Dem",1959,4,1
"1122","Netherlands",210,210,"Western Europe","Europe","Victor Marijnen","Victor Marijnen.Netherlands.1963.1964.Parliamentary Dem","Democracy","Parliamentary Dem",1963,2,1
"1123","Netherlands",210,210,"Western Europe","Europe","Joseph Cals","Joseph Cals.Netherlands.1965.1966.Parliamentary Dem","Democracy","Parliamentary Dem",1965,2,1
"1124","Netherlands",210,210,"Western Europe","Europe","Petrus de Jong","Petrus de Jong.Netherlands.1967.1970.Parliamentary Dem","Democracy","Parliamentary Dem",1967,4,1
"1125","Netherlands",210,210,"Western Europe","Europe","Barend Biesheuvel","Barend Biesheuvel.Netherlands.1971.1972.Parliamentary Dem","Democracy","Parliamentary Dem",1971,2,1
"1126","Netherlands",210,210,"Western Europe","Europe","Johannes den Uyl","Johannes den Uyl.Netherlands.1973.1976.Parliamentary Dem","Democracy","Parliamentary Dem",1973,4,1
"1127","Netherlands",210,210,"Western Europe","Europe","Andreas van Agt","Andreas van Agt.Netherlands.1977.1981.Parliamentary Dem","Democracy","Parliamentary Dem",1977,5,1
"1128","Netherlands",210,210,"Western Europe","Europe","Rudolphus Lubbers","Rudolphus Lubbers.Netherlands.1982.1993.Parliamentary Dem","Democracy","Parliamentary Dem",1982,12,1
"1129","Netherlands",210,210,"Western Europe","Europe","Wim Kok","Wim Kok.Netherlands.1994.2001.Parliamentary Dem","Democracy","Parliamentary Dem",1994,8,1
"1130","Netherlands",210,210,"Western Europe","Europe","Jan Peter Balkenende","Jan Peter Balkenende.Netherlands.2002.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2002,7,0
"1131","New Zealand",920,920,"Australia and New Zealand","Oceania","Peter Fraser","Peter Fraser.New Zealand.1946.1948.Parliamentary Dem","Democracy","Parliamentary Dem",1946,3,1
"1132","New Zealand",920,920,"Australia and New Zealand","Oceania","Sidney Holland","Sidney Holland.New Zealand.1949.1956.Parliamentary Dem","Democracy","Parliamentary Dem",1949,8,1
"1133","New Zealand",920,920,"Australia and New Zealand","Oceania","Walter Nash","Walter Nash.New Zealand.1957.1959.Parliamentary Dem","Democracy","Parliamentary Dem",1957,3,1
"1134","New Zealand",920,920,"Australia and New Zealand","Oceania","Keith Holyoake","Keith Holyoake.New Zealand.1960.1971.Parliamentary Dem","Democracy","Parliamentary Dem",1960,12,1
"1135","New Zealand",920,920,"Australia and New Zealand","Oceania","Norman Kirk","Norman Kirk.New Zealand.1972.1973.Parliamentary Dem","Democracy","Parliamentary Dem",1972,2,0
"1136","New Zealand",920,920,"Australia and New Zealand","Oceania","Wallace Rowling","Wallace Rowling.New Zealand.1974.1974.Parliamentary Dem","Democracy","Parliamentary Dem",1974,1,1
"1137","New Zealand",920,920,"Australia and New Zealand","Oceania","Robert Muldoon","Robert Muldoon.New Zealand.1975.1983.Parliamentary Dem","Democracy","Parliamentary Dem",1975,9,1
"1138","New Zealand",920,920,"Australia and New Zealand","Oceania","David Lange","David Lange.New Zealand.1984.1988.Parliamentary Dem","Democracy","Parliamentary Dem",1984,5,1
"1139","New Zealand",920,920,"Australia and New Zealand","Oceania","Geoffrey Palmer","Geoffrey Palmer.New Zealand.1989.1989.Parliamentary Dem","Democracy","Parliamentary Dem",1989,1,1
"1140","New Zealand",920,920,"Australia and New Zealand","Oceania","Jim Bolger","Jim Bolger.New Zealand.1990.1996.Parliamentary Dem","Democracy","Parliamentary Dem",1990,7,1
"1141","New Zealand",920,920,"Australia and New Zealand","Oceania","Jenny Shipley","Jenny Shipley.New Zealand.1997.1998.Parliamentary Dem","Democracy","Parliamentary Dem",1997,2,1
"1142","New Zealand",920,920,"Australia and New Zealand","Oceania","Helen Elizabeth Clark","Helen Elizabeth Clark.New Zealand.1999.2007.Parliamentary Dem","Democracy","Parliamentary Dem",1999,9,1
"1143","New Zealand",920,920,"Australia and New Zealand","Oceania","John Phillip Key","John Phillip Key.New Zealand.2008.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2008,1,0
"1144","Nicaragua",93,93,"Central America","Americas","Anastasio Somoza Garcia","Anastasio Somoza Garcia.Nicaragua.1946.1955.Military Dict","Non-democracy","Military Dict",1946,10,0
"1145","Nicaragua",93,93,"Central America","Americas","Luis Somoza Debayle","Luis Somoza Debayle.Nicaragua.1956.1966.Civilian Dict","Non-democracy","Civilian Dict",1956,11,1
"1146","Nicaragua",93,93,"Central America","Americas","Anastasio Somoza Debayle","Anastasio Somoza Debayle.Nicaragua.1967.1978.Military Dict","Non-democracy","Military Dict",1967,12,1
"1147","Nicaragua",93,93,"Central America","Americas","Daniel Ortega Saavedra","Daniel Ortega Saavedra.Nicaragua.1979.1983.Civilian Dict","Non-democracy","Civilian Dict",1979,5,1
"1148","Nicaragua",93,93,"Central America","Americas","Daniel Ortega Saavedra","Daniel Ortega Saavedra.Nicaragua.1984.1989.Presidential Dem","Democracy","Presidential Dem",1984,6,1
"1149","Nicaragua",93,93,"Central America","Americas","Violeta Barrios de Chamorro","Violeta Barrios de Chamorro.Nicaragua.1990.1996.Presidential Dem","Democracy","Presidential Dem",1990,7,1
"1150","Nicaragua",93,93,"Central America","Americas","Jose Arnoldo Aleman Lacayo","Jose Arnoldo Aleman Lacayo.Nicaragua.1997.2001.Presidential Dem","Democracy","Presidential Dem",1997,5,1
"1151","Nicaragua",93,93,"Central America","Americas","Enrique Bolanos Geyer","Enrique Bolanos Geyer.Nicaragua.2002.2006.Presidential Dem","Democracy","Presidential Dem",2002,5,1
"1152","Nicaragua",93,93,"Central America","Americas","Josï¿½ Daniel Ortega Saavedra","Josï¿½ Daniel Ortega Saavedra.Nicaragua.2007.2008.Presidential Dem","Democracy","Presidential Dem",2007,2,0
"1153","Niger",436,436,"Western Africa","Africa","Hamani Diori","Hamani Diori.Niger.1960.1973.Civilian Dict","Non-democracy","Civilian Dict",1960,14,1
"1154","Niger",436,436,"Western Africa","Africa","Seyni Kountche","Seyni Kountche.Niger.1974.1986.Military Dict","Non-democracy","Military Dict",1974,13,0
"1155","Niger",436,436,"Western Africa","Africa","Ali Saibou","Ali Saibou.Niger.1987.1992.Military Dict","Non-democracy","Military Dict",1987,6,1
"1156","Niger",436,436,"Western Africa","Africa","Mahamadou Issoufou","Mahamadou Issoufou.Niger.1993.1993.Mixed Dem","Democracy","Mixed Dem",1993,1,1
"1157","Niger",436,436,"Western Africa","Africa","Abdoulaye Souley","Abdoulaye Souley.Niger.1994.1994.Mixed Dem","Democracy","Mixed Dem",1994,1,1
"1158","Niger",436,436,"Western Africa","Africa","Hama Amadou","Hama Amadou.Niger.1995.1995.Mixed Dem","Democracy","Mixed Dem",1995,1,1
"1159","Niger",436,436,"Western Africa","Africa","Ibrahim Bare Mainassara","Ibrahim Bare Mainassara.Niger.1996.1998.Military Dict","Non-democracy","Military Dict",1996,3,0
"1160","Niger",436,436,"Western Africa","Africa","Dauoda Malam Wankï¿½","Dauoda Malam Wankï¿½.Niger.1999.1999.Military Dict","Non-democracy","Military Dict",1999,1,1
"1161","Niger",436,436,"Western Africa","Africa","Hama Amadou","Hama Amadou.Niger.2000.2006.Mixed Dem","Democracy","Mixed Dem",2000,7,1
"1162","Niger",436,436,"Western Africa","Africa","Seyni Oumarou","Seyni Oumarou.Niger.2007.2008.Mixed Dem","Democracy","Mixed Dem",2007,2,0
"1163","Nigeria",475,475,"Western Africa","Africa","Abubakar Tafawa Balewa","Abubakar Tafawa Balewa.Nigeria.1960.1965.Parliamentary Dem","Democracy","Parliamentary Dem",1960,6,0
"1164","Nigeria",475,475,"Western Africa","Africa","Yakubu Gowon","Yakubu Gowon.Nigeria.1966.1974.Military Dict","Non-democracy","Military Dict",1966,9,1
"1165","Nigeria",475,475,"Western Africa","Africa","Murtala Mohammed","Murtala Mohammed.Nigeria.1975.1975.Military Dict","Non-democracy","Military Dict",1975,1,0
"1166","Nigeria",475,475,"Western Africa","Africa","Olusegun Obasanjo","Olusegun Obasanjo.Nigeria.1976.1978.Military Dict","Non-democracy","Military Dict",1976,3,1
"1167","Nigeria",475,475,"Western Africa","Africa","Shehu Shagari","Shehu Shagari.Nigeria.1979.1982.Presidential Dem","Democracy","Presidential Dem",1979,4,1
"1168","Nigeria",475,475,"Western Africa","Africa","Mohammed Buhari","Mohammed Buhari.Nigeria.1983.1984.Military Dict","Non-democracy","Military Dict",1983,2,1
"1169","Nigeria",475,475,"Western Africa","Africa","Ibrahim Babangida","Ibrahim Babangida.Nigeria.1985.1992.Military Dict","Non-democracy","Military Dict",1985,8,1
"1170","Nigeria",475,475,"Western Africa","Africa","Sani Abacha","Sani Abacha.Nigeria.1993.1997.Military Dict","Non-democracy","Military Dict",1993,5,0
"1171","Nigeria",475,475,"Western Africa","Africa","Abdulsalami Abubakar","Abdulsalami Abubakar.Nigeria.1998.1998.Military Dict","Non-democracy","Military Dict",1998,1,1
"1172","Nigeria",475,475,"Western Africa","Africa","Olusegun Obasanjo","Olusegun Obasanjo.Nigeria.1999.2006.Presidential Dem","Democracy","Presidential Dem",1999,8,1
"1173","Nigeria",475,475,"Western Africa","Africa","Umaru Musa Yar'Adua","Umaru Musa Yar'Adua.Nigeria.2007.2008.Presidential Dem","Democracy","Presidential Dem",2007,2,0
"1174","North Korea",731,731,"Eastern Asia","Asia","Kim Il Sung","Kim Il Sung.North Korea.1948.1993.Military Dict","Non-democracy","Military Dict",1948,46,0
"1175","North Korea",731,731,"Eastern Asia","Asia","Kim Jong Il","Kim Jong Il.North Korea.1994.2008.Civilian Dict","Non-democracy","Civilian Dict",1994,15,0
"1176","Norway",385,385,"Northern Europe","Europe","Einar Gerhardsen","Einar Gerhardsen.Norway.1946.1950.Parliamentary Dem","Democracy","Parliamentary Dem",1946,5,1
"1177","Norway",385,385,"Northern Europe","Europe","Oscar Torp","Oscar Torp.Norway.1951.1954.Parliamentary Dem","Democracy","Parliamentary Dem",1951,4,1
"1178","Norway",385,385,"Northern Europe","Europe","Einar Gerhardsen","Einar Gerhardsen.Norway.1955.1964.Parliamentary Dem","Democracy","Parliamentary Dem",1955,10,1
"1179","Norway",385,385,"Northern Europe","Europe","Per Borten","Per Borten.Norway.1965.1970.Parliamentary Dem","Democracy","Parliamentary Dem",1965,6,1
"1180","Norway",385,385,"Northern Europe","Europe","Trygve Bratteli","Trygve Bratteli.Norway.1971.1971.Parliamentary Dem","Democracy","Parliamentary Dem",1971,1,1
"1181","Norway",385,385,"Northern Europe","Europe","Lars Korvald","Lars Korvald.Norway.1972.1972.Parliamentary Dem","Democracy","Parliamentary Dem",1972,1,1
"1182","Norway",385,385,"Northern Europe","Europe","Trygve Bratteli","Trygve Bratteli.Norway.1973.1975.Parliamentary Dem","Democracy","Parliamentary Dem",1973,3,1
"1183","Norway",385,385,"Northern Europe","Europe","Odvar Nordli","Odvar Nordli.Norway.1976.1980.Parliamentary Dem","Democracy","Parliamentary Dem",1976,5,1
"1184","Norway",385,385,"Northern Europe","Europe","Kare Isaachsen Willoch","Kare Isaachsen Willoch.Norway.1981.1985.Parliamentary Dem","Democracy","Parliamentary Dem",1981,5,1
"1185","Norway",385,385,"Northern Europe","Europe","Gro Harlem Brundtland","Gro Harlem Brundtland.Norway.1986.1988.Parliamentary Dem","Democracy","Parliamentary Dem",1986,3,1
"1186","Norway",385,385,"Northern Europe","Europe","Jan Peder Syse","Jan Peder Syse.Norway.1989.1989.Parliamentary Dem","Democracy","Parliamentary Dem",1989,1,1
"1187","Norway",385,385,"Northern Europe","Europe","Gro Harlem Brundtland","Gro Harlem Brundtland.Norway.1990.1995.Parliamentary Dem","Democracy","Parliamentary Dem",1990,6,1
"1188","Norway",385,385,"Northern Europe","Europe","Thorbjorn Jagland","Thorbjorn Jagland.Norway.1996.1996.Parliamentary Dem","Democracy","Parliamentary Dem",1996,1,1
"1189","Norway",385,385,"Northern Europe","Europe","Kjell Magne Bondevik","Kjell Magne Bondevik.Norway.1997.1999.Parliamentary Dem","Democracy","Parliamentary Dem",1997,3,1
"1190","Norway",385,385,"Northern Europe","Europe","Jena Stoltenberg","Jena Stoltenberg.Norway.2000.2000.Parliamentary Dem","Democracy","Parliamentary Dem",2000,1,1
"1191","Norway",385,385,"Northern Europe","Europe","Kjell Magne Bondevik","Kjell Magne Bondevik.Norway.2001.2004.Parliamentary Dem","Democracy","Parliamentary Dem",2001,4,1
"1192","Norway",385,385,"Northern Europe","Europe","Jens Stoltenberg","Jens Stoltenberg.Norway.2005.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2005,4,0
"1193","Oman",698,698,"Western Asia","Asia","Qabus ibn Sa'id","Qabus ibn Sa'id.Oman.1970.2008.Monarchy","Non-democracy","Monarchy",1970,39,0
"1194","Pakistan",770,769,"Southern Asia","Asia","Liaquat Ali Khan","Liaquat Ali Khan.Pakistan.1947.1950.Parliamentary Dem","Democracy","Parliamentary Dem",1947,4,0
"1195","Pakistan",770,769,"Southern Asia","Asia","Hwaja Nazim ad-Din","Hwaja Nazim ad-Din.Pakistan.1951.1952.Parliamentary Dem","Democracy","Parliamentary Dem",1951,2,1
"1196","Pakistan",770,769,"Southern Asia","Asia","Mohammad Ali Bogra","Mohammad Ali Bogra.Pakistan.1953.1954.Parliamentary Dem","Democracy","Parliamentary Dem",1953,2,1
"1197","Pakistan",770,769,"Southern Asia","Asia","Chauhdry Mohammad Ali","Chauhdry Mohammad Ali.Pakistan.1955.1955.Parliamentary Dem","Democracy","Parliamentary Dem",1955,1,1
"1198","Pakistan",770,769,"Southern Asia","Asia","Iskander Ali Mirza","Iskander Ali Mirza.Pakistan.1956.1957.Parliamentary Dem","Democracy","Parliamentary Dem",1956,2,1
"1199","Pakistan",770,769,"Southern Asia","Asia","Mohammad Ayub Khan","Mohammad Ayub Khan.Pakistan.1958.1968.Military Dict","Non-democracy","Military Dict",1958,11,1
"1200","Pakistan",770,769,"Southern Asia","Asia","Agha Mohammad Yahya Khan","Agha Mohammad Yahya Khan.Pakistan.1969.1970.Military Dict","Non-democracy","Military Dict",1969,2,1
"1201","Pakistan",770,769,"Southern Asia","Asia","Zulfikar Ali Bhutto","Zulfikar Ali Bhutto.Pakistan.1971.1971.Civilian Dict","Non-democracy","Civilian Dict",1971,1,1
"1202","Pakistan",770,770,"Southern Asia","Asia","Zulfikar Ali Bhutto","Zulfikar Ali Bhutto.Pakistan.1972.1976.Mixed Dem","Democracy","Mixed Dem",1972,5,1
"1203","Pakistan",770,770,"Southern Asia","Asia","Mohammad Zia-ul-Haq","Mohammad Zia-ul-Haq.Pakistan.1977.1987.Military Dict","Non-democracy","Military Dict",1977,11,0
"1204","Pakistan",770,770,"Southern Asia","Asia","Benazir Bhutto","Benazir Bhutto.Pakistan.1988.1989.Parliamentary Dem","Democracy","Parliamentary Dem",1988,2,1
"1205","Pakistan",770,770,"Southern Asia","Asia","Nawaz Sharif","Nawaz Sharif.Pakistan.1990.1992.Parliamentary Dem","Democracy","Parliamentary Dem",1990,3,1
"1206","Pakistan",770,770,"Southern Asia","Asia","Benazir Bhutto","Benazir Bhutto.Pakistan.1993.1995.Parliamentary Dem","Democracy","Parliamentary Dem",1993,3,1
"1207","Pakistan",770,770,"Southern Asia","Asia","Miraj Khalid","Miraj Khalid.Pakistan.1996.1996.Parliamentary Dem","Democracy","Parliamentary Dem",1996,1,1
"1208","Pakistan",770,770,"Southern Asia","Asia","Nawaz Sharif","Nawaz Sharif.Pakistan.1997.1998.Parliamentary Dem","Democracy","Parliamentary Dem",1997,2,1
"1209","Pakistan",770,770,"Southern Asia","Asia","Pervez Musharraf","Pervez Musharraf.Pakistan.1999.2007.Military Dict","Non-democracy","Military Dict",1999,9,1
"1210","Pakistan",770,770,"Southern Asia","Asia","Yousaf Raza Gilani","Yousaf Raza Gilani.Pakistan.2008.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2008,1,0
"1211","Palau",986,986,"Micronesia","Oceania","Kuniwo Nakamura","Kuniwo Nakamura.Palau.1994.2000.Presidential Dem","Democracy","Presidential Dem",1994,7,1
"1212","Palau",986,986,"Micronesia","Oceania","Tommy Esang Remengesau, Jr.","Tommy Esang Remengesau, Jr..Palau.2001.2008.Presidential Dem","Democracy","Presidential Dem",2001,8,0
"1213","Panama",95,95,"Central America","Americas","Enrique Jimenez","Enrique Jimenez.Panama.1946.1947.Civilian Dict","Non-democracy","Civilian Dict",1946,2,1
"1214","Panama",95,95,"Central America","Americas","Domingo Diaz Arosemena","Domingo Diaz Arosemena.Panama.1948.1948.Civilian Dict","Non-democracy","Civilian Dict",1948,1,1
"1215","Panama",95,95,"Central America","Americas","Arnulfo Arias Madrid","Arnulfo Arias Madrid.Panama.1949.1950.Presidential Dem","Democracy","Presidential Dem",1949,2,1
"1216","Panama",95,95,"Central America","Americas","military","military.Panama.1951.1951.Military Dict","Non-democracy","Military Dict",1951,1,1
"1217","Panama",95,95,"Central America","Americas","Jose Remon Cantero","Jose Remon Cantero.Panama.1952.1954.Presidential Dem","Democracy","Presidential Dem",1952,3,0
"1218","Panama",95,95,"Central America","Americas","Ricardo Arias Espinosa","Ricardo Arias Espinosa.Panama.1955.1955.Presidential Dem","Democracy","Presidential Dem",1955,1,1
"1219","Panama",95,95,"Central America","Americas","Ernesto de las Guardia","Ernesto de las Guardia.Panama.1956.1959.Presidential Dem","Democracy","Presidential Dem",1956,4,1
"1220","Panama",95,95,"Central America","Americas","Roberto Chiari","Roberto Chiari.Panama.1960.1963.Presidential Dem","Democracy","Presidential Dem",1960,4,1
"1221","Panama",95,95,"Central America","Americas","Marco Robles","Marco Robles.Panama.1964.1967.Presidential Dem","Democracy","Presidential Dem",1964,4,1
"1222","Panama",95,95,"Central America","Americas","Omar Torrijos Herrera","Omar Torrijos Herrera.Panama.1968.1980.Military Dict","Non-democracy","Military Dict",1968,13,0
"1223","Panama",95,95,"Central America","Americas","military","military.Panama.1981.1982.Military Dict","Non-democracy","Military Dict",1981,2,1
"1224","Panama",95,95,"Central America","Americas","Gen. Manuel Antonio Noriega Morena","Gen. Manuel Antonio Noriega Morena.Panama.1983.1988.Military Dict","Non-democracy","Military Dict",1983,6,1
"1225","Panama",95,95,"Central America","Americas","Guillermo Endara Galimany","Guillermo Endara Galimany.Panama.1989.1993.Presidential Dem","Democracy","Presidential Dem",1989,5,1
"1226","Panama",95,95,"Central America","Americas","Ernesto Perez Balladares","Ernesto Perez Balladares.Panama.1994.1998.Presidential Dem","Democracy","Presidential Dem",1994,5,1
"1227","Panama",95,95,"Central America","Americas","Mireya Elisa Moscoso de Arias","Mireya Elisa Moscoso de Arias.Panama.1999.2003.Presidential Dem","Democracy","Presidential Dem",1999,5,1
"1228","Panama",95,95,"Central America","Americas","Martï¿½n Erasto Torrijos Espino","Martï¿½n Erasto Torrijos Espino.Panama.2004.2008.Presidential Dem","Democracy","Presidential Dem",2004,5,0
"1229","Papua New Guinea",910,910,"Melanesia","Oceania","Michael Somare","Michael Somare.Papua New Guinea.1975.1979.Parliamentary Dem","Democracy","Parliamentary Dem",1975,5,1
"1230","Papua New Guinea",910,910,"Melanesia","Oceania","Julius Chan","Julius Chan.Papua New Guinea.1980.1981.Parliamentary Dem","Democracy","Parliamentary Dem",1980,2,1
"1231","Papua New Guinea",910,910,"Melanesia","Oceania","Michael Somare","Michael Somare.Papua New Guinea.1982.1984.Parliamentary Dem","Democracy","Parliamentary Dem",1982,3,1
"1232","Papua New Guinea",910,910,"Melanesia","Oceania","Paias Wingti","Paias Wingti.Papua New Guinea.1985.1987.Parliamentary Dem","Democracy","Parliamentary Dem",1985,3,1
"1233","Papua New Guinea",910,910,"Melanesia","Oceania","Rabbie Namaliu","Rabbie Namaliu.Papua New Guinea.1988.1991.Parliamentary Dem","Democracy","Parliamentary Dem",1988,4,1
"1234","Papua New Guinea",910,910,"Melanesia","Oceania","Paias Wingti","Paias Wingti.Papua New Guinea.1992.1993.Parliamentary Dem","Democracy","Parliamentary Dem",1992,2,1
"1235","Papua New Guinea",910,910,"Melanesia","Oceania","Julius Chan","Julius Chan.Papua New Guinea.1994.1996.Parliamentary Dem","Democracy","Parliamentary Dem",1994,3,1
"1236","Papua New Guinea",910,910,"Melanesia","Oceania","Bill Skate","Bill Skate.Papua New Guinea.1997.1998.Parliamentary Dem","Democracy","Parliamentary Dem",1997,2,1
"1237","Papua New Guinea",910,910,"Melanesia","Oceania","Mekere Morauta","Mekere Morauta.Papua New Guinea.1999.2001.Parliamentary Dem","Democracy","Parliamentary Dem",1999,3,1
"1238","Papua New Guinea",910,910,"Melanesia","Oceania","Michael Somare","Michael Somare.Papua New Guinea.2002.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2002,7,0
"1239","Paraguay",150,150,"South America","Americas","Higinio Morinigo","Higinio Morinigo.Paraguay.1946.1947.Military Dict","Non-democracy","Military Dict",1946,2,1
"1240","Paraguay",150,150,"South America","Americas","Juan Natalicio Gonzalez","Juan Natalicio Gonzalez.Paraguay.1948.1948.Civilian Dict","Non-democracy","Civilian Dict",1948,1,1
"1241","Paraguay",150,150,"South America","Americas","Federico Chavez","Federico Chavez.Paraguay.1949.1953.Civilian Dict","Non-democracy","Civilian Dict",1949,5,1
"1242","Paraguay",150,150,"South America","Americas","Alfredo Stroessner","Alfredo Stroessner.Paraguay.1954.1988.Military Dict","Non-democracy","Military Dict",1954,35,1
"1243","Paraguay",150,150,"South America","Americas","Andres Rodriguez","Andres Rodriguez.Paraguay.1989.1992.Presidential Dem","Democracy","Presidential Dem",1989,4,1
"1244","Paraguay",150,150,"South America","Americas","Juan Carlos Wasmosy","Juan Carlos Wasmosy.Paraguay.1993.1997.Presidential Dem","Democracy","Presidential Dem",1993,5,1
"1245","Paraguay",150,150,"South America","Americas","Raul Cuba Grau","Raul Cuba Grau.Paraguay.1998.1998.Presidential Dem","Democracy","Presidential Dem",1998,1,1
"1246","Paraguay",150,150,"South America","Americas","Luis Angel Gonzalez Macchi","Luis Angel Gonzalez Macchi.Paraguay.1999.2002.Presidential Dem","Democracy","Presidential Dem",1999,4,1
"1247","Paraguay",150,150,"South America","Americas","ï¿½scar Nicanor Duarte Frutos","ï¿½scar Nicanor Duarte Frutos.Paraguay.2003.2007.Presidential Dem","Democracy","Presidential Dem",2003,5,1
"1248","Paraguay",150,150,"South America","Americas","Fernando Armindo Lugo Mï¿½ndez","Fernando Armindo Lugo Mï¿½ndez.Paraguay.2008.2008.Presidential Dem","Democracy","Presidential Dem",2008,1,0
"1249","Peru",135,135,"South America","Americas","Jose Bustamente y Rivero","Jose Bustamente y Rivero.Peru.1946.1947.Presidential Dem","Democracy","Presidential Dem",1946,2,1
"1250","Peru",135,135,"South America","Americas","Manuel Odria","Manuel Odria.Peru.1948.1955.Military Dict","Non-democracy","Military Dict",1948,8,1
"1251","Peru",135,135,"South America","Americas","Manuel Prado y Ugarteche","Manuel Prado y Ugarteche.Peru.1956.1961.Presidential Dem","Democracy","Presidential Dem",1956,6,1
"1252","Peru",135,135,"South America","Americas","Ricardo Perez Godoy","Ricardo Perez Godoy.Peru.1962.1962.Military Dict","Non-democracy","Military Dict",1962,1,1
"1253","Peru",135,135,"South America","Americas","Fernando Belaunde Terry","Fernando Belaunde Terry.Peru.1963.1967.Presidential Dem","Democracy","Presidential Dem",1963,5,1
"1254","Peru",135,135,"South America","Americas","Juan Velasco Alvaredo","Juan Velasco Alvaredo.Peru.1968.1974.Military Dict","Non-democracy","Military Dict",1968,7,1
"1255","Peru",135,135,"South America","Americas","Francisco Morales Bermudez","Francisco Morales Bermudez.Peru.1975.1979.Military Dict","Non-democracy","Military Dict",1975,5,1
"1256","Peru",135,135,"South America","Americas","Fernando Belaunde Terry","Fernando Belaunde Terry.Peru.1980.1984.Presidential Dem","Democracy","Presidential Dem",1980,5,1
"1257","Peru",135,135,"South America","Americas","Alan Garcia Perez","Alan Garcia Perez.Peru.1985.1989.Presidential Dem","Democracy","Presidential Dem",1985,5,1
"1258","Peru",135,135,"South America","Americas","Alberto Fujimori","Alberto Fujimori.Peru.1990.1999.Civilian Dict","Non-democracy","Civilian Dict",1990,10,1
"1259","Peru",135,135,"South America","Americas","Valentin Paniagua Corazao","Valentin Paniagua Corazao.Peru.2000.2000.Civilian Dict","Non-democracy","Civilian Dict",2000,1,1
"1260","Peru",135,135,"South America","Americas","Alejandro Celestino Toledo Manrique","Alejandro Celestino Toledo Manrique.Peru.2001.2005.Presidential Dem","Democracy","Presidential Dem",2001,5,1
"1261","Peru",135,135,"South America","Americas","Alan Gabriel Ludwig Garcï¿½a Pï¿½rez","Alan Gabriel Ludwig Garcï¿½a Pï¿½rez.Peru.2006.2008.Presidential Dem","Democracy","Presidential Dem",2006,3,0
"1262","Philippines",840,840,"South-Eastern Asia","Asia","Manuel Roxas y Acuna","Manuel Roxas y Acuna.Philippines.1946.1947.Presidential Dem","Democracy","Presidential Dem",1946,2,0
"1263","Philippines",840,840,"South-Eastern Asia","Asia","Elpidio Quirino","Elpidio Quirino.Philippines.1948.1953.Presidential Dem","Democracy","Presidential Dem",1948,6,1
"1264","Philippines",840,840,"South-Eastern Asia","Asia","Ramon Magsaysay","Ramon Magsaysay.Philippines.1954.1956.Presidential Dem","Democracy","Presidential Dem",1954,3,0
"1265","Philippines",840,840,"South-Eastern Asia","Asia","Carlos Garcia","Carlos Garcia.Philippines.1957.1960.Presidential Dem","Democracy","Presidential Dem",1957,4,1
"1266","Philippines",840,840,"South-Eastern Asia","Asia","Diosdado Macapagal","Diosdado Macapagal.Philippines.1961.1964.Presidential Dem","Democracy","Presidential Dem",1961,4,1
"1267","Philippines",840,840,"South-Eastern Asia","Asia","Ferdinand Marcos","Ferdinand Marcos.Philippines.1965.1985.Civilian Dict","Non-democracy","Civilian Dict",1965,21,1
"1268","Philippines",840,840,"South-Eastern Asia","Asia","Corazon Aquino","Corazon Aquino.Philippines.1986.1991.Presidential Dem","Democracy","Presidential Dem",1986,6,1
"1269","Philippines",840,840,"South-Eastern Asia","Asia","Fidel Ramos","Fidel Ramos.Philippines.1992.1997.Presidential Dem","Democracy","Presidential Dem",1992,6,1
"1270","Philippines",840,840,"South-Eastern Asia","Asia","Joseph Estrada","Joseph Estrada.Philippines.1998.2000.Presidential Dem","Democracy","Presidential Dem",1998,3,1
"1271","Philippines",840,840,"South-Eastern Asia","Asia","Maria Gloria Macapagal","Maria Gloria Macapagal.Philippines.2001.2008.Presidential Dem","Democracy","Presidential Dem",2001,8,0
"1272","Poland",290,290,"Eastern Europe","Europe","Wladyslaw Gomulka","Wladyslaw Gomulka.Poland.1946.1947.Civilian Dict","Non-democracy","Civilian Dict",1946,2,1
"1273","Poland",290,290,"Eastern Europe","Europe","Boleslaw Bierut","Boleslaw Bierut.Poland.1948.1955.Civilian Dict","Non-democracy","Civilian Dict",1948,8,1
"1274","Poland",290,290,"Eastern Europe","Europe","Wladyslaw Gomulka","Wladyslaw Gomulka.Poland.1956.1969.Civilian Dict","Non-democracy","Civilian Dict",1956,14,1
"1275","Poland",290,290,"Eastern Europe","Europe","Edward Gierek","Edward Gierek.Poland.1970.1979.Civilian Dict","Non-democracy","Civilian Dict",1970,10,1
"1276","Poland",290,290,"Eastern Europe","Europe","Stanislaw Kania","Stanislaw Kania.Poland.1980.1980.Civilian Dict","Non-democracy","Civilian Dict",1980,1,1
"1277","Poland",290,290,"Eastern Europe","Europe","Wojciech Jaruzelski","Wojciech Jaruzelski.Poland.1981.1988.Military Dict","Non-democracy","Military Dict",1981,8,1
"1278","Poland",290,290,"Eastern Europe","Europe","Tadeusz Mazowiecki","Tadeusz Mazowiecki.Poland.1989.1990.Mixed Dem","Democracy","Mixed Dem",1989,2,1
"1279","Poland",290,290,"Eastern Europe","Europe","Jan Olszewski","Jan Olszewski.Poland.1991.1991.Mixed Dem","Democracy","Mixed Dem",1991,1,1
"1280","Poland",290,290,"Eastern Europe","Europe","Hana Suchocka","Hana Suchocka.Poland.1992.1992.Mixed Dem","Democracy","Mixed Dem",1992,1,1
"1281","Poland",290,290,"Eastern Europe","Europe","Waldemar Pawlak","Waldemar Pawlak.Poland.1993.1994.Mixed Dem","Democracy","Mixed Dem",1993,2,1
"1282","Poland",290,290,"Eastern Europe","Europe","Jozef Oleksy","Jozef Oleksy.Poland.1995.1995.Mixed Dem","Democracy","Mixed Dem",1995,1,1
"1283","Poland",290,290,"Eastern Europe","Europe","Wlodzimierz Cimoszewicz","Wlodzimierz Cimoszewicz.Poland.1996.1996.Mixed Dem","Democracy","Mixed Dem",1996,1,1
"1284","Poland",290,290,"Eastern Europe","Europe","Jerzy Buzek","Jerzy Buzek.Poland.1997.2000.Mixed Dem","Democracy","Mixed Dem",1997,4,1
"1285","Poland",290,290,"Eastern Europe","Europe","Leszek Miller","Leszek Miller.Poland.2001.2003.Mixed Dem","Democracy","Mixed Dem",2001,3,1
"1286","Poland",290,290,"Eastern Europe","Europe","Marek Belka","Marek Belka.Poland.2004.2004.Mixed Dem","Democracy","Mixed Dem",2004,1,1
"1287","Poland",290,290,"Eastern Europe","Europe","Kazimierz Marcinkiewicz","Kazimierz Marcinkiewicz.Poland.2005.2005.Mixed Dem","Democracy","Mixed Dem",2005,1,1
"1288","Poland",290,290,"Eastern Europe","Europe","Jaroslaw Kaczynski","Jaroslaw Kaczynski.Poland.2006.2006.Mixed Dem","Democracy","Mixed Dem",2006,1,1
"1289","Poland",290,290,"Eastern Europe","Europe","Donald Tusk","Donald Tusk.Poland.2007.2008.Mixed Dem","Democracy","Mixed Dem",2007,2,0
"1290","Portugal",235,235,"Southern Europe","Europe","Antonio de Oliveira Salazar","Antonio de Oliveira Salazar.Portugal.1946.1967.Civilian Dict","Non-democracy","Civilian Dict",1946,22,1
"1291","Portugal",235,235,"Southern Europe","Europe","Marcelo Caetano","Marcelo Caetano.Portugal.1968.1973.Civilian Dict","Non-democracy","Civilian Dict",1968,6,1
"1292","Portugal",235,235,"Southern Europe","Europe","Francisco da Costa Gomes","Francisco da Costa Gomes.Portugal.1974.1975.Military Dict","Non-democracy","Military Dict",1974,2,1
"1293","Portugal",235,235,"Southern Europe","Europe","Mario Alberto Nobre Lopes Soares","Mario Alberto Nobre Lopes Soares.Portugal.1976.1977.Mixed Dem","Democracy","Mixed Dem",1976,2,1
"1294","Portugal",235,235,"Southern Europe","Europe","Carlos da Mota Pinto","Carlos da Mota Pinto.Portugal.1978.1978.Mixed Dem","Democracy","Mixed Dem",1978,1,1
"1295","Portugal",235,235,"Southern Europe","Europe","Maria de Lourdes Pintasilgo","Maria de Lourdes Pintasilgo.Portugal.1979.1979.Mixed Dem","Democracy","Mixed Dem",1979,1,1
"1296","Portugal",235,235,"Southern Europe","Europe","Diogo Freitas do Amaral","Diogo Freitas do Amaral.Portugal.1980.1980.Mixed Dem","Democracy","Mixed Dem",1980,1,1
"1297","Portugal",235,235,"Southern Europe","Europe","Francisco Pinto Balsemao","Francisco Pinto Balsemao.Portugal.1981.1982.Mixed Dem","Democracy","Mixed Dem",1981,2,1
"1298","Portugal",235,235,"Southern Europe","Europe","Mario Soares","Mario Soares.Portugal.1983.1984.Mixed Dem","Democracy","Mixed Dem",1983,2,1
"1299","Portugal",235,235,"Southern Europe","Europe","Anibal Cavaco Silva","Anibal Cavaco Silva.Portugal.1985.1994.Mixed Dem","Democracy","Mixed Dem",1985,10,1
"1300","Portugal",235,235,"Southern Europe","Europe","Antonio Guterres","Antonio Guterres.Portugal.1995.2001.Mixed Dem","Democracy","Mixed Dem",1995,7,1
"1301","Portugal",235,235,"Southern Europe","Europe","Jose Manuel Durao Barroso","Jose Manuel Durao Barroso.Portugal.2002.2003.Mixed Dem","Democracy","Mixed Dem",2002,2,1
"1302","Portugal",235,235,"Southern Europe","Europe","Pedro (Miguel de) Santana Lopes","Pedro (Miguel de) Santana Lopes.Portugal.2004.2004.Mixed Dem","Democracy","Mixed Dem",2004,1,1
"1303","Portugal",235,235,"Southern Europe","Europe","Josï¿½ Sï¿½crates (Carvalho Pinto de Sousa)","Josï¿½ Sï¿½crates (Carvalho Pinto de Sousa).Portugal.2005.2008.Mixed Dem","Democracy","Mixed Dem",2005,4,0
"1304","Qatar",694,694,"Western Asia","Asia","Sheikh Ahmad ibn Ali Al Thani","Sheikh Ahmad ibn Ali Al Thani.Qatar.1971.1971.Monarchy","Non-democracy","Monarchy",1971,1,1
"1305","Qatar",694,694,"Western Asia","Asia","Sheikh Khalifah ibn Hamad Al Thani","Sheikh Khalifah ibn Hamad Al Thani.Qatar.1972.1994.Monarchy","Non-democracy","Monarchy",1972,23,1
"1306","Qatar",694,694,"Western Asia","Asia","Sheikh Hamad ibn Khalifah Al Thani","Sheikh Hamad ibn Khalifah Al Thani.Qatar.1995.2008.Monarchy","Non-democracy","Monarchy",1995,14,0
"1307","Romania",360,360,"Eastern Europe","Europe","Georghe Gheorghiu-Dej","Georghe Gheorghiu-Dej.Romania.1946.1953.Civilian Dict","Non-democracy","Civilian Dict",1946,8,1
"1308","Romania",360,360,"Eastern Europe","Europe","Gheorghe Apostol","Gheorghe Apostol.Romania.1954.1954.Civilian Dict","Non-democracy","Civilian Dict",1954,1,1
"1309","Romania",360,360,"Eastern Europe","Europe","Georghe Gheorghiu-Dej","Georghe Gheorghiu-Dej.Romania.1955.1964.Civilian Dict","Non-democracy","Civilian Dict",1955,10,0
"1310","Romania",360,360,"Eastern Europe","Europe","Nicolae Ceausescu","Nicolae Ceausescu.Romania.1965.1989.Civilian Dict","Non-democracy","Civilian Dict",1965,25,0
"1311","Romania",360,360,"Eastern Europe","Europe","Petre Roman","Petre Roman.Romania.1990.1990.Mixed Dem","Democracy","Mixed Dem",1990,1,1
"1312","Romania",360,360,"Eastern Europe","Europe","Theodor Stolojan","Theodor Stolojan.Romania.1991.1991.Mixed Dem","Democracy","Mixed Dem",1991,1,1
"1313","Romania",360,360,"Eastern Europe","Europe","Nicolae Vacaroiu","Nicolae Vacaroiu.Romania.1992.1995.Mixed Dem","Democracy","Mixed Dem",1992,4,1
"1314","Romania",360,360,"Eastern Europe","Europe","Victor Ciorbea","Victor Ciorbea.Romania.1996.1997.Mixed Dem","Democracy","Mixed Dem",1996,2,1
"1315","Romania",360,360,"Eastern Europe","Europe","Radu Vasile","Radu Vasile.Romania.1998.1998.Mixed Dem","Democracy","Mixed Dem",1998,1,1
"1316","Romania",360,360,"Eastern Europe","Europe","Mugur Isarescu","Mugur Isarescu.Romania.1999.1999.Mixed Dem","Democracy","Mixed Dem",1999,1,1
"1317","Romania",360,360,"Eastern Europe","Europe","Adrian Nastase","Adrian Nastase.Romania.2000.2003.Mixed Dem","Democracy","Mixed Dem",2000,4,1
"1318","Romania",360,360,"Eastern Europe","Europe","Calin Popescu-Tariceanu","Calin Popescu-Tariceanu.Romania.2004.2007.Mixed Dem","Democracy","Mixed Dem",2004,4,1
"1319","Romania",360,360,"Eastern Europe","Europe","Emil Boc","Emil Boc.Romania.2008.2008.Mixed Dem","Democracy","Mixed Dem",2008,1,0
"1320","Russian Federation",374,365,"Eastern Europe","Europe","Boris Yeltsin","Boris Yeltsin.Russian Federation.1991.1998.Civilian Dict","Non-democracy","Civilian Dict",1991,8,1
"1321","Russian Federation",374,365,"Eastern Europe","Europe","Vladimir Vladimirovich Putin","Vladimir Vladimirovich Putin.Russian Federation.1999.2007.Civilian Dict","Non-democracy","Civilian Dict",1999,9,1
"1322","Russian Federation",374,365,"Eastern Europe","Europe","Dmitry Anatolyevich Medvedev","Dmitry Anatolyevich Medvedev.Russian Federation.2008.2008.Civilian Dict","Non-democracy","Civilian Dict",2008,1,0
"1323","Rwanda",517,517,"Eastern Africa","Africa","Gregoire Kayibanda","Gregoire Kayibanda.Rwanda.1962.1972.Civilian Dict","Non-democracy","Civilian Dict",1962,11,1
"1324","Rwanda",517,517,"Eastern Africa","Africa","Juvenal Habyarimana","Juvenal Habyarimana.Rwanda.1973.1993.Military Dict","Non-democracy","Military Dict",1973,21,0
"1325","Rwanda",517,517,"Eastern Africa","Africa","Paul Kagame","Paul Kagame.Rwanda.1994.2008.Military Dict","Non-democracy","Military Dict",1994,15,0
"1326","Samoa",990,990,"Polynesia","Oceania","Malietoa Tanumafili II + Tupua Tamasese Mea'ole","Malietoa Tanumafili II + Tupua Tamasese Mea'ole.Samoa.1962.1962.Monarchy","Non-democracy","Monarchy",1962,1,0
"1327","Samoa",990,990,"Polynesia","Oceania","Malietoa Tanumafili II","Malietoa Tanumafili II.Samoa.1963.2006.Monarchy","Non-democracy","Monarchy",1963,44,0
"1328","Samoa",990,990,"Polynesia","Oceania","Tuiatua Tupua Tamasese Efi","Tuiatua Tupua Tamasese Efi.Samoa.2007.2008.Monarchy","Non-democracy","Monarchy",2007,2,0
"1329","San Marino",331,331,"Southern Europe","Europe","Romeo Morri + Marino Zanotti","Romeo Morri + Marino Zanotti.San Marino.1992.1992.Parliamentary Dem","Democracy","Parliamentary Dem",1992,1,1
"1330","San Marino",331,331,"Southern Europe","Europe","Gian Luigi Berti + Paride Andreoli","Gian Luigi Berti + Paride Andreoli.San Marino.1993.1993.Parliamentary Dem","Democracy","Parliamentary Dem",1993,1,1
"1331","San Marino",331,331,"Southern Europe","Europe","Renzo Ghiotti + Luciano Ciavatta","Renzo Ghiotti + Luciano Ciavatta.San Marino.1994.1994.Parliamentary Dem","Democracy","Parliamentary Dem",1994,1,1
"1332","San Marino",331,331,"Southern Europe","Europe","Pier Natalino Mularoni + Marino Venturini","Pier Natalino Mularoni + Marino Venturini.San Marino.1995.1995.Parliamentary Dem","Democracy","Parliamentary Dem",1995,1,1
"1333","San Marino",331,331,"Southern Europe","Europe","Maurizio Rattini + Giancarlo Venturini","Maurizio Rattini + Giancarlo Venturini.San Marino.1996.1996.Parliamentary Dem","Democracy","Parliamentary Dem",1996,1,1
"1334","San Marino",331,331,"Southern Europe","Europe","Luigi Mazza + Marino Zanotti","Luigi Mazza + Marino Zanotti.San Marino.1997.1997.Parliamentary Dem","Democracy","Parliamentary Dem",1997,1,1
"1335","San Marino",331,331,"Southern Europe","Europe","Pietro Berti + Paolo Bollini","Pietro Berti + Paolo Bollini.San Marino.1998.1998.Parliamentary Dem","Democracy","Parliamentary Dem",1998,1,1
"1336","San Marino",331,331,"Southern Europe","Europe","Marino Bollini + Giuseppe Arzilli","Marino Bollini + Giuseppe Arzilli.San Marino.1999.1999.Parliamentary Dem","Democracy","Parliamentary Dem",1999,1,1
"1337","San Marino",331,331,"Southern Europe","Europe","Gian Franco Terenzi + Enzo Colombini","Gian Franco Terenzi + Enzo Colombini.San Marino.2000.2000.Parliamentary Dem","Democracy","Parliamentary Dem",2000,1,1
"1338","San Marino",331,331,"Southern Europe","Europe","Alberto Cecchetti + Gino Giovagnoli","Alberto Cecchetti + Gino Giovagnoli.San Marino.2001.2001.Parliamentary Dem","Democracy","Parliamentary Dem",2001,1,1
"1339","San Marino",331,331,"Southern Europe","Europe","Giuseppe Maria Morganti + Mauro Chiaruzzi","Giuseppe Maria Morganti + Mauro Chiaruzzi.San Marino.2002.2002.Parliamentary Dem","Democracy","Parliamentary Dem",2002,1,1
"1340","San Marino",331,331,"Southern Europe","Europe","Giovanni Lonfernini + Valeria Ciavatta","Giovanni Lonfernini + Valeria Ciavatta.San Marino.2003.2003.Parliamentary Dem","Democracy","Parliamentary Dem",2003,1,1
"1341","San Marino",331,331,"Southern Europe","Europe","Giuseppe Arzilli + Roberto Raschi","Giuseppe Arzilli + Roberto Raschi.San Marino.2004.2004.Parliamentary Dem","Democracy","Parliamentary Dem",2004,1,1
"1342","San Marino",331,331,"Southern Europe","Europe","Claudio Muccioli + Antonello Bacciocchi","Claudio Muccioli + Antonello Bacciocchi.San Marino.2005.2005.Parliamentary Dem","Democracy","Parliamentary Dem",2005,1,1
"1343","San Marino",331,331,"Southern Europe","Europe","Antonio Carattoni + Roberto Giorgetti","Antonio Carattoni + Roberto Giorgetti.San Marino.2006.2006.Parliamentary Dem","Democracy","Parliamentary Dem",2006,1,1
"1344","San Marino",331,331,"Southern Europe","Europe","Mirko Tomassoni + Alberto Selva","Mirko Tomassoni + Alberto Selva.San Marino.2007.2007.Parliamentary Dem","Democracy","Parliamentary Dem",2007,1,1
"1345","San Marino",331,331,"Southern Europe","Europe","Ernesto Benedettini + Assunta Meloni","Ernesto Benedettini + Assunta Meloni.San Marino.2008.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2008,1,0
"1346","Sao Tome and Principe",403,403,"Middle Africa","Africa","Manuel Pinto da Costa","Manuel Pinto da Costa.Sao Tome and Principe.1975.1990.Civilian Dict","Non-democracy","Civilian Dict",1975,16,1
"1347","Sao Tome and Principe",403,403,"Middle Africa","Africa","Daniel Lima dos Santos Daio","Daniel Lima dos Santos Daio.Sao Tome and Principe.1991.1991.Mixed Dem","Democracy","Mixed Dem",1991,1,1
"1348","Sao Tome and Principe",403,403,"Middle Africa","Africa","Norberto Jose d'Alva Costa Alegre","Norberto Jose d'Alva Costa Alegre.Sao Tome and Principe.1992.1993.Mixed Dem","Democracy","Mixed Dem",1992,2,1
"1349","Sao Tome and Principe",403,403,"Middle Africa","Africa","Carlos da Graca","Carlos da Graca.Sao Tome and Principe.1994.1994.Mixed Dem","Democracy","Mixed Dem",1994,1,1
"1350","Sao Tome and Principe",403,403,"Middle Africa","Africa","Armindo Vaz d'Almeida","Armindo Vaz d'Almeida.Sao Tome and Principe.1995.1995.Mixed Dem","Democracy","Mixed Dem",1995,1,1
"1351","Sao Tome and Principe",403,403,"Middle Africa","Africa","Raul Braganca Neto","Raul Braganca Neto.Sao Tome and Principe.1996.1998.Mixed Dem","Democracy","Mixed Dem",1996,3,1
"1352","Sao Tome and Principe",403,403,"Middle Africa","Africa","Guilherme Posser da Costa","Guilherme Posser da Costa.Sao Tome and Principe.1999.2000.Mixed Dem","Democracy","Mixed Dem",1999,2,1
"1353","Sao Tome and Principe",403,403,"Middle Africa","Africa","Evaristo Carvalho","Evaristo Carvalho.Sao Tome and Principe.2001.2001.Mixed Dem","Democracy","Mixed Dem",2001,1,1
"1354","Sao Tome and Principe",403,403,"Middle Africa","Africa","Maria das Neves Ceita Baptista de Sousa","Maria das Neves Ceita Baptista de Sousa.Sao Tome and Principe.2002.2003.Mixed Dem","Democracy","Mixed Dem",2002,2,1
"1355","Sao Tome and Principe",403,403,"Middle Africa","Africa","Damiï¿½o Vaz d'Almeida","Damiï¿½o Vaz d'Almeida.Sao Tome and Principe.2004.2004.Mixed Dem","Democracy","Mixed Dem",2004,1,1
"1356","Sao Tome and Principe",403,403,"Middle Africa","Africa","Maria do Carmo Silveira","Maria do Carmo Silveira.Sao Tome and Principe.2005.2005.Mixed Dem","Democracy","Mixed Dem",2005,1,1
"1357","Sao Tome and Principe",403,403,"Middle Africa","Africa","Tomï¿½ Vera Cruz","Tomï¿½ Vera Cruz.Sao Tome and Principe.2006.2007.Mixed Dem","Democracy","Mixed Dem",2006,2,1
"1358","Sao Tome and Principe",403,403,"Middle Africa","Africa","Joaquim Rafael Branco","Joaquim Rafael Branco.Sao Tome and Principe.2008.2008.Mixed Dem","Democracy","Mixed Dem",2008,1,0
"1359","Saudi Arabia",670,670,"Western Asia","Asia","Ibn Sa'ud","Ibn Sa'ud.Saudi Arabia.1946.1952.Monarchy","Non-democracy","Monarchy",1946,7,0
"1360","Saudi Arabia",670,670,"Western Asia","Asia","Sa'ud","Sa'ud.Saudi Arabia.1953.1963.Monarchy","Non-democracy","Monarchy",1953,11,1
"1361","Saudi Arabia",670,670,"Western Asia","Asia","Faysal","Faysal.Saudi Arabia.1964.1974.Monarchy","Non-democracy","Monarchy",1964,11,0
"1362","Saudi Arabia",670,670,"Western Asia","Asia","Khalid","Khalid.Saudi Arabia.1975.1981.Monarchy","Non-democracy","Monarchy",1975,7,0
"1363","Saudi Arabia",670,670,"Western Asia","Asia","Fahd","Fahd.Saudi Arabia.1982.1995.Monarchy","Non-democracy","Monarchy",1982,14,1
"1364","Saudi Arabia",670,670,"Western Asia","Asia","Abdallah","Abdallah.Saudi Arabia.1996.2008.Monarchy","Non-democracy","Monarchy",1996,13,0
"1365","Senegal",433,433,"Western Africa","Africa","Leopold Senghor","Leopold Senghor.Senegal.1960.1980.Civilian Dict","Non-democracy","Civilian Dict",1960,21,1
"1366","Senegal",433,433,"Western Africa","Africa","Abdou Diouf","Abdou Diouf.Senegal.1981.1999.Civilian Dict","Non-democracy","Civilian Dict",1981,19,1
"1367","Senegal",433,433,"Western Africa","Africa","Moustapha Niasse","Moustapha Niasse.Senegal.2000.2000.Mixed Dem","Democracy","Mixed Dem",2000,1,1
"1368","Senegal",433,433,"Western Africa","Africa","Mame Madior Boye","Mame Madior Boye.Senegal.2001.2001.Mixed Dem","Democracy","Mixed Dem",2001,1,1
"1369","Senegal",433,433,"Western Africa","Africa","Idrissa Seck","Idrissa Seck.Senegal.2002.2003.Mixed Dem","Democracy","Mixed Dem",2002,2,1
"1370","Senegal",433,433,"Western Africa","Africa","Macky Sall","Macky Sall.Senegal.2004.2006.Mixed Dem","Democracy","Mixed Dem",2004,3,1
"1371","Senegal",433,433,"Western Africa","Africa","Hadjibou Soumarï¿½","Hadjibou Soumarï¿½.Senegal.2007.2008.Mixed Dem","Democracy","Mixed Dem",2007,2,0
"1372","Serbia",342,342,"Southern Europe","Europe","Vojislav Kostunica","Vojislav Kostunica.Serbia.2006.2007.Mixed Dem","Democracy","Mixed Dem",2006,2,1
"1373","Serbia",342,342,"Southern Europe","Europe","Mirko Cvetkovic","Mirko Cvetkovic.Serbia.2008.2008.Mixed Dem","Democracy","Mixed Dem",2008,1,0
"1374","Serbia and Montenegro",347,347,"Southern Europe","Europe","Slobodan Milosevic","Slobodan Milosevic.Serbia and Montenegro.1991.1999.Civilian Dict","Non-democracy","Civilian Dict",1991,9,1
"1375","Serbia and Montenegro",347,347,"Southern Europe","Europe","Milomir Minic","Milomir Minic.Serbia and Montenegro.2000.2000.Mixed Dem","Democracy","Mixed Dem",2000,1,1
"1376","Serbia and Montenegro",347,347,"Southern Europe","Europe","Zoran Djindjic","Zoran Djindjic.Serbia and Montenegro.2001.2002.Mixed Dem","Democracy","Mixed Dem",2001,2,0
"1377","Serbia and Montenegro",347,347,"Southern Europe","Europe","Zoran Zivkovic","Zoran Zivkovic.Serbia and Montenegro.2003.2003.Mixed Dem","Democracy","Mixed Dem",2003,1,1
"1378","Serbia and Montenegro",347,347,"Southern Europe","Europe","Vojislav Kostunica","Vojislav Kostunica.Serbia and Montenegro.2004.2005.Mixed Dem","Democracy","Mixed Dem",2004,2,0
"1379","Seychelles",591,591,"Eastern Africa","Africa","James Mancham","James Mancham.Seychelles.1976.1976.Civilian Dict","Non-democracy","Civilian Dict",1976,1,1
"1380","Seychelles",591,591,"Eastern Africa","Africa","Albert Rene","Albert Rene.Seychelles.1977.2003.Civilian Dict","Non-democracy","Civilian Dict",1977,27,1
"1381","Seychelles",591,591,"Eastern Africa","Africa","James Alix Michel","James Alix Michel.Seychelles.2004.2008.Civilian Dict","Non-democracy","Civilian Dict",2004,5,0
"1382","Sierra Leone",451,451,"Western Africa","Africa","Milton Margai","Milton Margai.Sierra Leone.1961.1963.Parliamentary Dem","Democracy","Parliamentary Dem",1961,3,0
"1383","Sierra Leone",451,451,"Western Africa","Africa","Albert Margai","Albert Margai.Sierra Leone.1964.1966.Parliamentary Dem","Democracy","Parliamentary Dem",1964,3,1
"1384","Sierra Leone",451,451,"Western Africa","Africa","Andrew Juxon-Smith","Andrew Juxon-Smith.Sierra Leone.1967.1967.Military Dict","Non-democracy","Military Dict",1967,1,1
"1385","Sierra Leone",451,451,"Western Africa","Africa","Siaka Stevens","Siaka Stevens.Sierra Leone.1968.1984.Civilian Dict","Non-democracy","Civilian Dict",1968,17,1
"1386","Sierra Leone",451,451,"Western Africa","Africa","Joseph Saidu Momoh","Joseph Saidu Momoh.Sierra Leone.1985.1991.Military Dict","Non-democracy","Military Dict",1985,7,1
"1387","Sierra Leone",451,451,"Western Africa","Africa","Valentine Strasser","Valentine Strasser.Sierra Leone.1992.1995.Military Dict","Non-democracy","Military Dict",1992,4,1
"1388","Sierra Leone",451,451,"Western Africa","Africa","Ahmad Tejan Kabbah","Ahmad Tejan Kabbah.Sierra Leone.1996.1996.Presidential Dem","Democracy","Presidential Dem",1996,1,1
"1389","Sierra Leone",451,451,"Western Africa","Africa","Johnny Paul Koroma","Johnny Paul Koroma.Sierra Leone.1997.1997.Military Dict","Non-democracy","Military Dict",1997,1,1
"1390","Sierra Leone",451,451,"Western Africa","Africa","Ahmad Tejan Kabbah","Ahmad Tejan Kabbah.Sierra Leone.1998.2006.Presidential Dem","Democracy","Presidential Dem",1998,9,1
"1391","Sierra Leone",451,451,"Western Africa","Africa","Ernest Bai Koroma","Ernest Bai Koroma.Sierra Leone.2007.2008.Presidential Dem","Democracy","Presidential Dem",2007,2,0
"1392","Singapore",830,830,"South-Eastern Asia","Asia","Lee Kuan Yew","Lee Kuan Yew.Singapore.1965.1989.Civilian Dict","Non-democracy","Civilian Dict",1965,25,1
"1393","Singapore",830,830,"South-Eastern Asia","Asia","Goh Chok Tong","Goh Chok Tong.Singapore.1990.2003.Civilian Dict","Non-democracy","Civilian Dict",1990,14,1
"1394","Singapore",830,830,"South-Eastern Asia","Asia","Lee Hsien Loong","Lee Hsien Loong.Singapore.2004.2008.Military Dict","Non-democracy","Military Dict",2004,5,0
"1395","Slovakia",317,317,"Eastern Europe","Europe","Vladimir Meciar","Vladimir Meciar.Slovakia.1993.1997.Parliamentary Dem","Democracy","Parliamentary Dem",1993,5,1
"1396","Slovakia",317,317,"Eastern Europe","Europe","Milkulis Dzurinda","Milkulis Dzurinda.Slovakia.1998.1998.Parliamentary Dem","Democracy","Parliamentary Dem",1998,1,1
"1397","Slovakia",317,317,"Eastern Europe","Europe","Milkulis Dzurinda","Milkulis Dzurinda.Slovakia.1999.2005.Mixed Dem","Democracy","Mixed Dem",1999,7,1
"1398","Slovakia",317,317,"Eastern Europe","Europe","Robert Fico","Robert Fico.Slovakia.2006.2008.Mixed Dem","Democracy","Mixed Dem",2006,3,0
"1399","Slovenia",349,349,"Southern Europe","Europe","Lojze Peterle","Lojze Peterle.Slovenia.1991.1991.Mixed Dem","Democracy","Mixed Dem",1991,1,1
"1400","Slovenia",349,349,"Southern Europe","Europe","Janez Drnovsek","Janez Drnovsek.Slovenia.1992.2001.Mixed Dem","Democracy","Mixed Dem",1992,10,1
"1401","Slovenia",349,349,"Southern Europe","Europe","Anton Rop","Anton Rop.Slovenia.2002.2002.Mixed Dem","Democracy","Mixed Dem",2002,1,1
"1402","Slovenia",349,349,"Southern Europe","Europe","Anton Rop","Anton Rop.Slovenia.2003.2003.Parliamentary Dem","Democracy","Parliamentary Dem",2003,1,1
"1403","Slovenia",349,349,"Southern Europe","Europe","Janez Jansa","Janez Jansa.Slovenia.2004.2007.Parliamentary Dem","Democracy","Parliamentary Dem",2004,4,1
"1404","Slovenia",349,349,"Southern Europe","Europe","Borut Pahor","Borut Pahor.Slovenia.2008.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2008,1,0
"1405","Solomon Islands",940,940,"Melanesia","Oceania","Peter Kenilorea","Peter Kenilorea.Solomon Islands.1978.1980.Parliamentary Dem","Democracy","Parliamentary Dem",1978,3,1
"1406","Solomon Islands",940,940,"Melanesia","Oceania","Solomon Mamaloni","Solomon Mamaloni.Solomon Islands.1981.1983.Parliamentary Dem","Democracy","Parliamentary Dem",1981,3,1
"1407","Solomon Islands",940,940,"Melanesia","Oceania","Peter Kenilorea","Peter Kenilorea.Solomon Islands.1984.1985.Parliamentary Dem","Democracy","Parliamentary Dem",1984,2,1
"1408","Solomon Islands",940,940,"Melanesia","Oceania","Ezekiel Alebua","Ezekiel Alebua.Solomon Islands.1986.1988.Parliamentary Dem","Democracy","Parliamentary Dem",1986,3,1
"1409","Solomon Islands",940,940,"Melanesia","Oceania","Solomon Mamaloni","Solomon Mamaloni.Solomon Islands.1989.1992.Parliamentary Dem","Democracy","Parliamentary Dem",1989,4,1
"1410","Solomon Islands",940,940,"Melanesia","Oceania","Francis Billy Hilly","Francis Billy Hilly.Solomon Islands.1993.1996.Parliamentary Dem","Democracy","Parliamentary Dem",1993,4,1
"1411","Solomon Islands",940,940,"Melanesia","Oceania","Bartholomew Ulufa'alu","Bartholomew Ulufa'alu.Solomon Islands.1997.1999.Parliamentary Dem","Democracy","Parliamentary Dem",1997,3,1
"1412","Solomon Islands",940,940,"Melanesia","Oceania","Mannasseh Sogavare","Mannasseh Sogavare.Solomon Islands.2000.2000.Parliamentary Dem","Democracy","Parliamentary Dem",2000,1,1
"1413","Solomon Islands",940,940,"Melanesia","Oceania","Allan Kemakeza","Allan Kemakeza.Solomon Islands.2001.2005.Parliamentary Dem","Democracy","Parliamentary Dem",2001,5,1
"1414","Solomon Islands",940,940,"Melanesia","Oceania","Manasseh Damukana Sogavare","Manasseh Damukana Sogavare.Solomon Islands.2006.2006.Parliamentary Dem","Democracy","Parliamentary Dem",2006,1,1
"1415","Solomon Islands",940,940,"Melanesia","Oceania","Derek Sikua","Derek Sikua.Solomon Islands.2007.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2007,2,0
"1416","Somalia",520,520,"Eastern Africa","Africa","Abdirashid Shermarke","Abdirashid Shermarke.Somalia.1960.1963.Parliamentary Dem","Democracy","Parliamentary Dem",1960,4,0
"1417","Somalia",520,520,"Eastern Africa","Africa","Abdirizak Hussain","Abdirizak Hussain.Somalia.1964.1966.Parliamentary Dem","Democracy","Parliamentary Dem",1964,3,1
"1418","Somalia",520,520,"Eastern Africa","Africa","Mohammed Egal","Mohammed Egal.Somalia.1967.1968.Parliamentary Dem","Democracy","Parliamentary Dem",1967,2,1
"1419","Somalia",520,520,"Eastern Africa","Africa","Mohammed Siyad Barreh","Mohammed Siyad Barreh.Somalia.1969.1990.Military Dict","Non-democracy","Military Dict",1969,22,1
"1420","Somalia",520,520,"Eastern Africa","Africa","Ali Mahdi Muhammad","Ali Mahdi Muhammad.Somalia.1991.1996.Civilian Dict","Non-democracy","Civilian Dict",1991,6,1
"1421","Somalia",520,520,"Eastern Africa","Africa","Five person collective chairmanship of National Salvation Council including previous president, Ali Mahdi Muhammad, Osman Hassan Ali Ato, Abdulkadir Muhammad Aden Zoppo, Abdullahi Yussuf Ahmad, and Aden Abdullahi Nur Gabiyow","Five person collective chairmanship of National Salvation Council including previous president, Ali Mahdi Muhammad, Osman Hassan Ali Ato, Abdulkadir Muhammad Aden Zoppo, Abdullahi Yussuf Ahmad, and Aden Abdullahi Nur Gabiyow.Somalia.1997.1999.Civilian Dict","Non-democracy","Civilian Dict",1997,3,1
"1422","Somalia",520,520,"Eastern Africa","Africa","Abdiqassim Salad Hassan","Abdiqassim Salad Hassan.Somalia.2000.2003.Civilian Dict","Non-democracy","Civilian Dict",2000,4,1
"1423","Somalia",520,520,"Eastern Africa","Africa","Abdullahi Yussuf Ahmad","Abdullahi Yussuf Ahmad.Somalia.2004.2006.Civilian Dict","Non-democracy","Civilian Dict",2004,3,1
"1424","Somalia",520,520,"Eastern Africa","Africa","Aden Muhammad Nur Madobe","Aden Muhammad Nur Madobe.Somalia.2007.2008.Civilian Dict","Non-democracy","Civilian Dict",2007,2,0
"1425","South Africa",560,560,"Southern Africa","Africa","Jan Christian Smuts","Jan Christian Smuts.South Africa.1946.1947.Civilian Dict","Non-democracy","Civilian Dict",1946,2,1
"1426","South Africa",560,560,"Southern Africa","Africa","Daniel Malan","Daniel Malan.South Africa.1948.1953.Civilian Dict","Non-democracy","Civilian Dict",1948,6,1
"1427","South Africa",560,560,"Southern Africa","Africa","Johannes Strijdom","Johannes Strijdom.South Africa.1954.1957.Civilian Dict","Non-democracy","Civilian Dict",1954,4,0
"1428","South Africa",560,560,"Southern Africa","Africa","Hendrik Verwoerd","Hendrik Verwoerd.South Africa.1958.1965.Civilian Dict","Non-democracy","Civilian Dict",1958,8,0
"1429","South Africa",560,560,"Southern Africa","Africa","B. Johannes Vorster","B. Johannes Vorster.South Africa.1966.1977.Civilian Dict","Non-democracy","Civilian Dict",1966,12,1
"1430","South Africa",560,560,"Southern Africa","Africa","Pieter Botha","Pieter Botha.South Africa.1978.1988.Civilian Dict","Non-democracy","Civilian Dict",1978,11,1
"1431","South Africa",560,560,"Southern Africa","Africa","Frederik de Klerk","Frederik de Klerk.South Africa.1989.1993.Civilian Dict","Non-democracy","Civilian Dict",1989,5,1
"1432","South Africa",560,560,"Southern Africa","Africa","Nelson Mandela","Nelson Mandela.South Africa.1994.1998.Civilian Dict","Non-democracy","Civilian Dict",1994,5,1
"1433","South Africa",560,560,"Southern Africa","Africa","Thabo Mbeki","Thabo Mbeki.South Africa.1999.2007.Civilian Dict","Non-democracy","Civilian Dict",1999,9,1
"1434","South Africa",560,560,"Southern Africa","Africa","Kgalema Petrus Motlanthe","Kgalema Petrus Motlanthe.South Africa.2008.2008.Civilian Dict","Non-democracy","Civilian Dict",2008,1,0
"1435","South Korea",732,732,"Eastern Asia","Asia","Syngman Rhee","Syngman Rhee.South Korea.1948.1959.Civilian Dict","Non-democracy","Civilian Dict",1948,12,1
"1436","South Korea",732,732,"Eastern Asia","Asia","John Myun Chang","John Myun Chang.South Korea.1960.1960.Parliamentary Dem","Democracy","Parliamentary Dem",1960,1,1
"1437","South Korea",732,732,"Eastern Asia","Asia","Park Chung Hee","Park Chung Hee.South Korea.1961.1978.Military Dict","Non-democracy","Military Dict",1961,18,0
"1438","South Korea",732,732,"Eastern Asia","Asia","Chun Doo Hwan","Chun Doo Hwan.South Korea.1979.1987.Military Dict","Non-democracy","Military Dict",1979,9,1
"1439","South Korea",732,732,"Eastern Asia","Asia","Roh Tae Woo","Roh Tae Woo.South Korea.1988.1992.Presidential Dem","Democracy","Presidential Dem",1988,5,1
"1440","South Korea",732,732,"Eastern Asia","Asia","Kim Young Sam","Kim Young Sam.South Korea.1993.1997.Presidential Dem","Democracy","Presidential Dem",1993,5,1
"1441","South Korea",732,732,"Eastern Asia","Asia","Kim Dae Jung","Kim Dae Jung.South Korea.1998.2002.Presidential Dem","Democracy","Presidential Dem",1998,5,1
"1442","South Korea",732,732,"Eastern Asia","Asia","Roh Moo Hyun","Roh Moo Hyun.South Korea.2003.2007.Presidential Dem","Democracy","Presidential Dem",2003,5,1
"1443","South Korea",732,732,"Eastern Asia","Asia","Lee Myung Bak","Lee Myung Bak.South Korea.2008.2008.Presidential Dem","Democracy","Presidential Dem",2008,1,1
"1444","Spain",230,230,"Southern Europe","Europe","Francisco Franco y Bahamonde","Francisco Franco y Bahamonde.Spain.1946.1974.Military Dict","Non-democracy","Military Dict",1946,29,0
"1445","Spain",230,230,"Southern Europe","Europe","Carlos Arias Navarro","Carlos Arias Navarro.Spain.1975.1975.Civilian Dict","Non-democracy","Civilian Dict",1975,1,1
"1446","Spain",230,230,"Southern Europe","Europe","Adolfo Suarez Gonzalez","Adolfo Suarez Gonzalez.Spain.1976.1976.Civilian Dict","Non-democracy","Civilian Dict",1976,1,1
"1447","Spain",230,230,"Southern Europe","Europe","Adolfo Suarez Gonzalez","Adolfo Suarez Gonzalez.Spain.1977.1980.Parliamentary Dem","Democracy","Parliamentary Dem",1977,4,1
"1448","Spain",230,230,"Southern Europe","Europe","Leopoldo Calvo-Sotelo y Bustelo","Leopoldo Calvo-Sotelo y Bustelo.Spain.1981.1981.Parliamentary Dem","Democracy","Parliamentary Dem",1981,1,1
"1449","Spain",230,230,"Southern Europe","Europe","Felipe Gonzalez Marquez","Felipe Gonzalez Marquez.Spain.1982.1995.Parliamentary Dem","Democracy","Parliamentary Dem",1982,14,1
"1450","Spain",230,230,"Southern Europe","Europe","Jose Maria Aznar","Jose Maria Aznar.Spain.1996.2003.Parliamentary Dem","Democracy","Parliamentary Dem",1996,8,1
"1451","Spain",230,230,"Southern Europe","Europe","Josï¿½ Luis Rodrï¿½guez Zapatero","Josï¿½ Luis Rodrï¿½guez Zapatero.Spain.2004.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2004,5,0
"1452","Sri Lanka",780,780,"Southern Asia","Asia","Don Stephen Senanayake","Don Stephen Senanayake.Sri Lanka.1948.1951.Parliamentary Dem","Democracy","Parliamentary Dem",1948,4,0
"1453","Sri Lanka",780,780,"Southern Asia","Asia","Dudley Shelton Senanayake","Dudley Shelton Senanayake.Sri Lanka.1952.1952.Parliamentary Dem","Democracy","Parliamentary Dem",1952,1,1
"1454","Sri Lanka",780,780,"Southern Asia","Asia","John Lionel Kotelawala","John Lionel Kotelawala.Sri Lanka.1953.1955.Parliamentary Dem","Democracy","Parliamentary Dem",1953,3,1
"1455","Sri Lanka",780,780,"Southern Asia","Asia","S.W.R.D. Bandaranaike","S.W.R.D. Bandaranaike.Sri Lanka.1956.1958.Parliamentary Dem","Democracy","Parliamentary Dem",1956,3,0
"1456","Sri Lanka",780,780,"Southern Asia","Asia","Vijayananda Dahanayake","Vijayananda Dahanayake.Sri Lanka.1959.1959.Parliamentary Dem","Democracy","Parliamentary Dem",1959,1,1
"1457","Sri Lanka",780,780,"Southern Asia","Asia","Sirimavo Bandaranaike","Sirimavo Bandaranaike.Sri Lanka.1960.1964.Parliamentary Dem","Democracy","Parliamentary Dem",1960,5,1
"1458","Sri Lanka",780,780,"Southern Asia","Asia","Dudley Shelton Senanayake","Dudley Shelton Senanayake.Sri Lanka.1965.1969.Parliamentary Dem","Democracy","Parliamentary Dem",1965,5,1
"1459","Sri Lanka",780,780,"Southern Asia","Asia","Sirimavo Bandaranaike","Sirimavo Bandaranaike.Sri Lanka.1970.1976.Parliamentary Dem","Democracy","Parliamentary Dem",1970,7,1
"1460","Sri Lanka",780,780,"Southern Asia","Asia","Junius Richard Jayawardene","Junius Richard Jayawardene.Sri Lanka.1977.1988.Civilian Dict","Non-democracy","Civilian Dict",1977,12,1
"1461","Sri Lanka",780,780,"Southern Asia","Asia","Dingiri Banda Wijetunge","Dingiri Banda Wijetunge.Sri Lanka.1989.1992.Presidential Dem","Democracy","Presidential Dem",1989,4,1
"1462","Sri Lanka",780,780,"Southern Asia","Asia","Ranil Wickremasinghe","Ranil Wickremasinghe.Sri Lanka.1993.1993.Presidential Dem","Democracy","Presidential Dem",1993,1,1
"1463","Sri Lanka",780,780,"Southern Asia","Asia","Sirimavo Bandaranaike","Sirimavo Bandaranaike.Sri Lanka.1994.1999.Presidential Dem","Democracy","Presidential Dem",1994,6,0
"1464","Sri Lanka",780,780,"Southern Asia","Asia","Ratnasiri Wickremanayake","Ratnasiri Wickremanayake.Sri Lanka.2000.2000.Presidential Dem","Democracy","Presidential Dem",2000,1,1
"1465","Sri Lanka",780,780,"Southern Asia","Asia","Ranil Wickremasinghe","Ranil Wickremasinghe.Sri Lanka.2001.2003.Presidential Dem","Democracy","Presidential Dem",2001,3,1
"1466","Sri Lanka",780,780,"Southern Asia","Asia","Mahinda Rajapakse","Mahinda Rajapakse.Sri Lanka.2004.2004.Presidential Dem","Democracy","Presidential Dem",2004,1,1
"1467","Sri Lanka",780,780,"Southern Asia","Asia","Ratnasiri Wickremanayake","Ratnasiri Wickremanayake.Sri Lanka.2005.2008.Presidential Dem","Democracy","Presidential Dem",2005,4,0
"1468","St. Kitts and Nevis",60,60,"Caribbean","Americas","Kennedy Simmonds","Kennedy Simmonds.St. Kitts and Nevis.1983.1994.Parliamentary Dem","Democracy","Parliamentary Dem",1983,12,1
"1469","St. Kitts and Nevis",60,60,"Caribbean","Americas","Denzil Douglas","Denzil Douglas.St. Kitts and Nevis.1995.2008.Parliamentary Dem","Democracy","Parliamentary Dem",1995,14,0
"1470","St. Lucia",56,56,"Caribbean","Americas","Allan Louisy","Allan Louisy.St. Lucia.1979.1980.Parliamentary Dem","Democracy","Parliamentary Dem",1979,2,1
"1471","St. Lucia",56,56,"Caribbean","Americas","Winston Francis Cenac","Winston Francis Cenac.St. Lucia.1981.1981.Parliamentary Dem","Democracy","Parliamentary Dem",1981,1,1
"1472","St. Lucia",56,56,"Caribbean","Americas","John Compton","John Compton.St. Lucia.1982.1995.Parliamentary Dem","Democracy","Parliamentary Dem",1982,14,1
"1473","St. Lucia",56,56,"Caribbean","Americas","Vaughan Allen Lewis","Vaughan Allen Lewis.St. Lucia.1996.1996.Parliamentary Dem","Democracy","Parliamentary Dem",1996,1,1
"1474","St. Lucia",56,56,"Caribbean","Americas","Kenny Anthony","Kenny Anthony.St. Lucia.1997.2005.Parliamentary Dem","Democracy","Parliamentary Dem",1997,9,1
"1475","St. Lucia",56,56,"Caribbean","Americas","John Compton","John Compton.St. Lucia.2006.2006.Parliamentary Dem","Democracy","Parliamentary Dem",2006,1,1
"1476","St. Lucia",56,56,"Caribbean","Americas","Stephenson King","Stephenson King.St. Lucia.2007.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2007,2,0
"1477","St. Vincent and the Grenadines",57,57,"Caribbean","Americas","Milton Cato","Milton Cato.St. Vincent and the Grenadines.1979.1983.Parliamentary Dem","Democracy","Parliamentary Dem",1979,5,1
"1478","St. Vincent and the Grenadines",57,57,"Caribbean","Americas","James Fitz-Allen Mitchell","James Fitz-Allen Mitchell.St. Vincent and the Grenadines.1984.1999.Parliamentary Dem","Democracy","Parliamentary Dem",1984,16,1
"1479","St. Vincent and the Grenadines",57,57,"Caribbean","Americas","Arnhim Eustace","Arnhim Eustace.St. Vincent and the Grenadines.2000.2000.Parliamentary Dem","Democracy","Parliamentary Dem",2000,1,1
"1480","St. Vincent and the Grenadines",57,57,"Caribbean","Americas","Ralph Gonsalves","Ralph Gonsalves.St. Vincent and the Grenadines.2001.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2001,8,0
"1481","Sudan",625,625,"Northern Africa","Africa","Abdullah Khalil","Abdullah Khalil.Sudan.1956.1957.Parliamentary Dem","Democracy","Parliamentary Dem",1956,2,1
"1482","Sudan",625,625,"Northern Africa","Africa","Ibrahim Abboud","Ibrahim Abboud.Sudan.1958.1963.Military Dict","Non-democracy","Military Dict",1958,6,1
"1483","Sudan",625,625,"Northern Africa","Africa","Sir al-Khatim al-Khalifah","Sir al-Khatim al-Khalifah.Sudan.1964.1964.Civilian Dict","Non-democracy","Civilian Dict",1964,1,1
"1484","Sudan",625,625,"Northern Africa","Africa","Mohammed Mahgoub","Mohammed Mahgoub.Sudan.1965.1965.Parliamentary Dem","Democracy","Parliamentary Dem",1965,1,1
"1485","Sudan",625,625,"Northern Africa","Africa","Sadiq al-Mahdi","Sadiq al-Mahdi.Sudan.1966.1966.Parliamentary Dem","Democracy","Parliamentary Dem",1966,1,1
"1486","Sudan",625,625,"Northern Africa","Africa","Mohammed Mahgoub","Mohammed Mahgoub.Sudan.1967.1968.Parliamentary Dem","Democracy","Parliamentary Dem",1967,2,1
"1487","Sudan",625,625,"Northern Africa","Africa","Jaafar el-Nemery","Jaafar el-Nemery.Sudan.1969.1984.Military Dict","Non-democracy","Military Dict",1969,16,1
"1488","Sudan",625,625,"Northern Africa","Africa","Abdul Rahman Swareddabhab","Abdul Rahman Swareddabhab.Sudan.1985.1985.Military Dict","Non-democracy","Military Dict",1985,1,1
"1489","Sudan",625,625,"Northern Africa","Africa","Sadiq al-Mahdi","Sadiq al-Mahdi.Sudan.1986.1988.Parliamentary Dem","Democracy","Parliamentary Dem",1986,3,1
"1490","Sudan",625,625,"Northern Africa","Africa","Omar Hassan Ahmad al-Bashir","Omar Hassan Ahmad al-Bashir.Sudan.1989.2008.Military Dict","Non-democracy","Military Dict",1989,20,0
"1491","Suriname",115,115,"South America","Americas","Henck Arron","Henck Arron.Suriname.1975.1979.Parliamentary Dem","Democracy","Parliamentary Dem",1975,5,1
"1492","Suriname",115,115,"South America","Americas","Desi Bouterse","Desi Bouterse.Suriname.1980.1987.Military Dict","Non-democracy","Military Dict",1980,8,1
"1493","Suriname",115,115,"South America","Americas","Ramsewak Shankar","Ramsewak Shankar.Suriname.1988.1989.Presidential Dem","Democracy","Presidential Dem",1988,2,1
"1494","Suriname",115,115,"South America","Americas","Desi Bouterse","Desi Bouterse.Suriname.1990.1990.Military Dict","Non-democracy","Military Dict",1990,1,1
"1495","Suriname",115,115,"South America","Americas","Ronald Venetiaan","Ronald Venetiaan.Suriname.1991.1995.Presidential Dem","Democracy","Presidential Dem",1991,5,1
"1496","Suriname",115,115,"South America","Americas","Jules Albert Wijdenbosch","Jules Albert Wijdenbosch.Suriname.1996.1999.Presidential Dem","Democracy","Presidential Dem",1996,4,1
"1497","Suriname",115,115,"South America","Americas","Ronald Venetiaan","Ronald Venetiaan.Suriname.2000.2008.Presidential Dem","Democracy","Presidential Dem",2000,9,0
"1498","Swaziland",572,572,"Southern Africa","Africa","Sobhuza II","Sobhuza II.Swaziland.1968.1981.Monarchy","Non-democracy","Monarchy",1968,14,0
"1499","Swaziland",572,572,"Southern Africa","Africa","Queen Dzeliwe","Queen Dzeliwe.Swaziland.1982.1982.Monarchy","Non-democracy","Monarchy",1982,1,1
"1500","Swaziland",572,572,"Southern Africa","Africa","Queen Ntombi","Queen Ntombi.Swaziland.1983.1985.Monarchy","Non-democracy","Monarchy",1983,3,1
"1501","Swaziland",572,572,"Southern Africa","Africa","Mswati III","Mswati III.Swaziland.1986.2008.Monarchy","Non-democracy","Monarchy",1986,23,0
"1502","Sweden",380,380,"Northern Europe","Europe","Tage Erlander","Tage Erlander.Sweden.1946.1968.Parliamentary Dem","Democracy","Parliamentary Dem",1946,23,1
"1503","Sweden",380,380,"Northern Europe","Europe","Olof Palme","Olof Palme.Sweden.1969.1975.Parliamentary Dem","Democracy","Parliamentary Dem",1969,7,1
"1504","Sweden",380,380,"Northern Europe","Europe","Thorbjorn Falldin","Thorbjorn Falldin.Sweden.1976.1977.Parliamentary Dem","Democracy","Parliamentary Dem",1976,2,1
"1505","Sweden",380,380,"Northern Europe","Europe","Ola Ullsten","Ola Ullsten.Sweden.1978.1978.Parliamentary Dem","Democracy","Parliamentary Dem",1978,1,1
"1506","Sweden",380,380,"Northern Europe","Europe","Thorbjorn Falldin","Thorbjorn Falldin.Sweden.1979.1981.Parliamentary Dem","Democracy","Parliamentary Dem",1979,3,1
"1507","Sweden",380,380,"Northern Europe","Europe","Olof Palme","Olof Palme.Sweden.1982.1985.Parliamentary Dem","Democracy","Parliamentary Dem",1982,4,0
"1508","Sweden",380,380,"Northern Europe","Europe","Ingvar Carlsson","Ingvar Carlsson.Sweden.1986.1990.Parliamentary Dem","Democracy","Parliamentary Dem",1986,5,1
"1509","Sweden",380,380,"Northern Europe","Europe","Carl Bildt","Carl Bildt.Sweden.1991.1993.Parliamentary Dem","Democracy","Parliamentary Dem",1991,3,1
"1510","Sweden",380,380,"Northern Europe","Europe","Ingvar Carlsson","Ingvar Carlsson.Sweden.1994.1995.Parliamentary Dem","Democracy","Parliamentary Dem",1994,2,1
"1511","Sweden",380,380,"Northern Europe","Europe","Goran Persson","Goran Persson.Sweden.1996.2005.Parliamentary Dem","Democracy","Parliamentary Dem",1996,10,1
"1512","Sweden",380,380,"Northern Europe","Europe","John Fredrik Reinfeldt","John Fredrik Reinfeldt.Sweden.2006.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2006,3,0
"1513","Switzerland",225,225,"Western Europe","Europe","Karl Kobelt","Karl Kobelt.Switzerland.1946.1946.Presidential Dem","Democracy","Presidential Dem",1946,1,1
"1514","Switzerland",225,225,"Western Europe","Europe","Philipp Etter","Philipp Etter.Switzerland.1947.1947.Presidential Dem","Democracy","Presidential Dem",1947,1,1
"1515","Switzerland",225,225,"Western Europe","Europe","Enrico Celio","Enrico Celio.Switzerland.1948.1948.Presidential Dem","Democracy","Presidential Dem",1948,1,1
"1516","Switzerland",225,225,"Western Europe","Europe","Ernst Nobs","Ernst Nobs.Switzerland.1949.1949.Presidential Dem","Democracy","Presidential Dem",1949,1,1
"1517","Switzerland",225,225,"Western Europe","Europe","Max-Edouard Petitpierre","Max-Edouard Petitpierre.Switzerland.1950.1950.Presidential Dem","Democracy","Presidential Dem",1950,1,1
"1518","Switzerland",225,225,"Western Europe","Europe","Adolf Eduard von Steiger","Adolf Eduard von Steiger.Switzerland.1951.1951.Presidential Dem","Democracy","Presidential Dem",1951,1,1
"1519","Switzerland",225,225,"Western Europe","Europe","Karl Kobelt","Karl Kobelt.Switzerland.1952.1952.Presidential Dem","Democracy","Presidential Dem",1952,1,1
"1520","Switzerland",225,225,"Western Europe","Europe","Philipp Etter","Philipp Etter.Switzerland.1953.1953.Presidential Dem","Democracy","Presidential Dem",1953,1,1
"1521","Switzerland",225,225,"Western Europe","Europe","Rodolphe Rubattel","Rodolphe Rubattel.Switzerland.1954.1954.Presidential Dem","Democracy","Presidential Dem",1954,1,1
"1522","Switzerland",225,225,"Western Europe","Europe","Max-Edouard Petitpierre","Max-Edouard Petitpierre.Switzerland.1955.1955.Presidential Dem","Democracy","Presidential Dem",1955,1,1
"1523","Switzerland",225,225,"Western Europe","Europe","Markus Feldmann","Markus Feldmann.Switzerland.1956.1956.Presidential Dem","Democracy","Presidential Dem",1956,1,1
"1524","Switzerland",225,225,"Western Europe","Europe","Hans Streuli","Hans Streuli.Switzerland.1957.1957.Presidential Dem","Democracy","Presidential Dem",1957,1,1
"1525","Switzerland",225,225,"Western Europe","Europe","Thomas Emil Leo Holenstein","Thomas Emil Leo Holenstein.Switzerland.1958.1958.Presidential Dem","Democracy","Presidential Dem",1958,1,1
"1526","Switzerland",225,225,"Western Europe","Europe","Paul Chaudet","Paul Chaudet.Switzerland.1959.1959.Presidential Dem","Democracy","Presidential Dem",1959,1,1
"1527","Switzerland",225,225,"Western Europe","Europe","Max-Edouard Petitpierre","Max-Edouard Petitpierre.Switzerland.1960.1960.Presidential Dem","Democracy","Presidential Dem",1960,1,1
"1528","Switzerland",225,225,"Western Europe","Europe","Friedrich Traugott Wahlen","Friedrich Traugott Wahlen.Switzerland.1961.1961.Presidential Dem","Democracy","Presidential Dem",1961,1,1
"1529","Switzerland",225,225,"Western Europe","Europe","Paul Chaudet","Paul Chaudet.Switzerland.1962.1962.Presidential Dem","Democracy","Presidential Dem",1962,1,1
"1530","Switzerland",225,225,"Western Europe","Europe","Willy Spuhler","Willy Spuhler.Switzerland.1963.1963.Presidential Dem","Democracy","Presidential Dem",1963,1,1
"1531","Switzerland",225,225,"Western Europe","Europe","Ludwig von Moos","Ludwig von Moos.Switzerland.1964.1964.Presidential Dem","Democracy","Presidential Dem",1964,1,1
"1532","Switzerland",225,225,"Western Europe","Europe","Hans-Peter Tschudi","Hans-Peter Tschudi.Switzerland.1965.1965.Presidential Dem","Democracy","Presidential Dem",1965,1,1
"1533","Switzerland",225,225,"Western Europe","Europe","Hans Schaffner","Hans Schaffner.Switzerland.1966.1966.Presidential Dem","Democracy","Presidential Dem",1966,1,1
"1534","Switzerland",225,225,"Western Europe","Europe","Roger Bonvin","Roger Bonvin.Switzerland.1967.1967.Presidential Dem","Democracy","Presidential Dem",1967,1,1
"1535","Switzerland",225,225,"Western Europe","Europe","Willy Spuhler","Willy Spuhler.Switzerland.1968.1968.Presidential Dem","Democracy","Presidential Dem",1968,1,1
"1536","Switzerland",225,225,"Western Europe","Europe","Ludwig von Moos","Ludwig von Moos.Switzerland.1969.1969.Presidential Dem","Democracy","Presidential Dem",1969,1,1
"1537","Switzerland",225,225,"Western Europe","Europe","Hans-Peter Tschudi","Hans-Peter Tschudi.Switzerland.1970.1970.Presidential Dem","Democracy","Presidential Dem",1970,1,1
"1538","Switzerland",225,225,"Western Europe","Europe","Rudolf Gnagi","Rudolf Gnagi.Switzerland.1971.1971.Presidential Dem","Democracy","Presidential Dem",1971,1,1
"1539","Switzerland",225,225,"Western Europe","Europe","Nello Celio","Nello Celio.Switzerland.1972.1972.Presidential Dem","Democracy","Presidential Dem",1972,1,1
"1540","Switzerland",225,225,"Western Europe","Europe","Roger Bonvin","Roger Bonvin.Switzerland.1973.1973.Presidential Dem","Democracy","Presidential Dem",1973,1,1
"1541","Switzerland",225,225,"Western Europe","Europe","Ernst Brugger","Ernst Brugger.Switzerland.1974.1974.Presidential Dem","Democracy","Presidential Dem",1974,1,1
"1542","Switzerland",225,225,"Western Europe","Europe","Pierre Graber","Pierre Graber.Switzerland.1975.1975.Presidential Dem","Democracy","Presidential Dem",1975,1,1
"1543","Switzerland",225,225,"Western Europe","Europe","Rudolf Gnagi","Rudolf Gnagi.Switzerland.1976.1976.Presidential Dem","Democracy","Presidential Dem",1976,1,1
"1544","Switzerland",225,225,"Western Europe","Europe","Kurt Furgler","Kurt Furgler.Switzerland.1977.1977.Presidential Dem","Democracy","Presidential Dem",1977,1,1
"1545","Switzerland",225,225,"Western Europe","Europe","Willi Ritschard","Willi Ritschard.Switzerland.1978.1978.Presidential Dem","Democracy","Presidential Dem",1978,1,1
"1546","Switzerland",225,225,"Western Europe","Europe","Hans Hurlimann","Hans Hurlimann.Switzerland.1979.1979.Presidential Dem","Democracy","Presidential Dem",1979,1,1
"1547","Switzerland",225,225,"Western Europe","Europe","Georges-Andre Chevallaz","Georges-Andre Chevallaz.Switzerland.1980.1980.Presidential Dem","Democracy","Presidential Dem",1980,1,1
"1548","Switzerland",225,225,"Western Europe","Europe","Kurt Furgler","Kurt Furgler.Switzerland.1981.1981.Presidential Dem","Democracy","Presidential Dem",1981,1,1
"1549","Switzerland",225,225,"Western Europe","Europe","Fritz Honegger","Fritz Honegger.Switzerland.1982.1982.Presidential Dem","Democracy","Presidential Dem",1982,1,1
"1550","Switzerland",225,225,"Western Europe","Europe","Pierre Aubert","Pierre Aubert.Switzerland.1983.1983.Presidential Dem","Democracy","Presidential Dem",1983,1,1
"1551","Switzerland",225,225,"Western Europe","Europe","Leon Schlumpf","Leon Schlumpf.Switzerland.1984.1984.Presidential Dem","Democracy","Presidential Dem",1984,1,1
"1552","Switzerland",225,225,"Western Europe","Europe","Kurt Furgler","Kurt Furgler.Switzerland.1985.1985.Presidential Dem","Democracy","Presidential Dem",1985,1,1
"1553","Switzerland",225,225,"Western Europe","Europe","Alphons Egli","Alphons Egli.Switzerland.1986.1986.Presidential Dem","Democracy","Presidential Dem",1986,1,1
"1554","Switzerland",225,225,"Western Europe","Europe","Pierre Aubert","Pierre Aubert.Switzerland.1987.1987.Presidential Dem","Democracy","Presidential Dem",1987,1,1
"1555","Switzerland",225,225,"Western Europe","Europe","Otto Stich","Otto Stich.Switzerland.1988.1988.Presidential Dem","Democracy","Presidential Dem",1988,1,1
"1556","Switzerland",225,225,"Western Europe","Europe","Jean-Pascal Delamuraz","Jean-Pascal Delamuraz.Switzerland.1989.1989.Presidential Dem","Democracy","Presidential Dem",1989,1,1
"1557","Switzerland",225,225,"Western Europe","Europe","Arnold Koller","Arnold Koller.Switzerland.1990.1990.Presidential Dem","Democracy","Presidential Dem",1990,1,1
"1558","Switzerland",225,225,"Western Europe","Europe","Flavio Cotti","Flavio Cotti.Switzerland.1991.1991.Presidential Dem","Democracy","Presidential Dem",1991,1,1
"1559","Switzerland",225,225,"Western Europe","Europe","Rene Felber","Rene Felber.Switzerland.1992.1992.Presidential Dem","Democracy","Presidential Dem",1992,1,1
"1560","Switzerland",225,225,"Western Europe","Europe","Adolf Ogi","Adolf Ogi.Switzerland.1993.1993.Presidential Dem","Democracy","Presidential Dem",1993,1,1
"1561","Switzerland",225,225,"Western Europe","Europe","Otto Stich","Otto Stich.Switzerland.1994.1994.Presidential Dem","Democracy","Presidential Dem",1994,1,1
"1562","Switzerland",225,225,"Western Europe","Europe","Kaspar Villiger","Kaspar Villiger.Switzerland.1995.1995.Presidential Dem","Democracy","Presidential Dem",1995,1,1
"1563","Switzerland",225,225,"Western Europe","Europe","Jean-Pascal Delamuraz","Jean-Pascal Delamuraz.Switzerland.1996.1996.Presidential Dem","Democracy","Presidential Dem",1996,1,1
"1564","Switzerland",225,225,"Western Europe","Europe","Arnold Koller","Arnold Koller.Switzerland.1997.1997.Presidential Dem","Democracy","Presidential Dem",1997,1,1
"1565","Switzerland",225,225,"Western Europe","Europe","Flavio Cotti","Flavio Cotti.Switzerland.1998.1998.Presidential Dem","Democracy","Presidential Dem",1998,1,1
"1566","Switzerland",225,225,"Western Europe","Europe","Ruth Dreifuss","Ruth Dreifuss.Switzerland.1999.1999.Presidential Dem","Democracy","Presidential Dem",1999,1,1
"1567","Switzerland",225,225,"Western Europe","Europe","Adolf Ogi","Adolf Ogi.Switzerland.2000.2000.Presidential Dem","Democracy","Presidential Dem",2000,1,1
"1568","Switzerland",225,225,"Western Europe","Europe","Moritz Leuenberger","Moritz Leuenberger.Switzerland.2001.2001.Presidential Dem","Democracy","Presidential Dem",2001,1,1
"1569","Switzerland",225,225,"Western Europe","Europe","Kaspar Villiger","Kaspar Villiger.Switzerland.2002.2002.Presidential Dem","Democracy","Presidential Dem",2002,1,1
"1570","Switzerland",225,225,"Western Europe","Europe","Pascal Couchepin","Pascal Couchepin.Switzerland.2003.2003.Presidential Dem","Democracy","Presidential Dem",2003,1,1
"1571","Switzerland",225,225,"Western Europe","Europe","Joseph Deiss","Joseph Deiss.Switzerland.2004.2004.Presidential Dem","Democracy","Presidential Dem",2004,1,1
"1572","Switzerland",225,225,"Western Europe","Europe","Samuel Schmid","Samuel Schmid.Switzerland.2005.2005.Presidential Dem","Democracy","Presidential Dem",2005,1,1
"1573","Switzerland",225,225,"Western Europe","Europe","Moritz Leuenberger","Moritz Leuenberger.Switzerland.2006.2006.Presidential Dem","Democracy","Presidential Dem",2006,1,1
"1574","Switzerland",225,225,"Western Europe","Europe","Micheline Calmy-Rey","Micheline Calmy-Rey.Switzerland.2007.2007.Presidential Dem","Democracy","Presidential Dem",2007,1,1
"1575","Switzerland",225,225,"Western Europe","Europe","Pascal Couchepin","Pascal Couchepin.Switzerland.2008.2008.Presidential Dem","Democracy","Presidential Dem",2008,1,0
"1576","Syria",652,652,"Western Asia","Asia","Shukri al-Kuwatli","Shukri al-Kuwatli.Syria.1946.1948.Civilian Dict","Non-democracy","Civilian Dict",1946,3,1
"1577","Syria",652,652,"Western Asia","Asia","Adib ash-Shishakli","Adib ash-Shishakli.Syria.1949.1953.Military Dict","Non-democracy","Military Dict",1949,5,1
"1578","Syria",652,652,"Western Asia","Asia","military","military.Syria.1954.1954.Military Dict","Non-democracy","Military Dict",1954,1,1
"1579","Syria",652,652,"Western Asia","Asia","Shukri al-Kuwatli","Shukri al-Kuwatli.Syria.1955.1960.Civilian Dict","Non-democracy","Civilian Dict",1955,6,1
"1580","Syria",652,652,"Western Asia","Asia","Nazim al-Kudsi","Nazim al-Kudsi.Syria.1961.1962.Civilian Dict","Non-democracy","Civilian Dict",1961,2,1
"1581","Syria",652,652,"Western Asia","Asia","Amin al-Hafez","Amin al-Hafez.Syria.1963.1965.Military Dict","Non-democracy","Military Dict",1963,3,1
"1582","Syria",652,652,"Western Asia","Asia","Salah al-Jadid","Salah al-Jadid.Syria.1966.1969.Military Dict","Non-democracy","Military Dict",1966,4,1
"1583","Syria",652,652,"Western Asia","Asia","Hafez al-Assad","Hafez al-Assad.Syria.1970.1999.Military Dict","Non-democracy","Military Dict",1970,30,0
"1584","Syria",652,652,"Western Asia","Asia","Bashar al-Assad","Bashar al-Assad.Syria.2000.2008.Military Dict","Non-democracy","Military Dict",2000,9,0
"1585","Taiwan",713,713,"Eastern Asia","Asia","Chiang Kai-shek","Chiang Kai-shek.Taiwan.1949.1974.Military Dict","Non-democracy","Military Dict",1949,26,0
"1586","Taiwan",713,713,"Eastern Asia","Asia","Yen Chia-kan","Yen Chia-kan.Taiwan.1975.1977.Civilian Dict","Non-democracy","Civilian Dict",1975,3,1
"1587","Taiwan",713,713,"Eastern Asia","Asia","Chiang Ching-kuo","Chiang Ching-kuo.Taiwan.1978.1987.Civilian Dict","Non-democracy","Civilian Dict",1978,10,0
"1588","Taiwan",713,713,"Eastern Asia","Asia","Lee Teng-hui","Lee Teng-hui.Taiwan.1988.1995.Civilian Dict","Non-democracy","Civilian Dict",1988,8,1
"1589","Taiwan",713,713,"Eastern Asia","Asia","Lien Chan","Lien Chan.Taiwan.1996.1996.Mixed Dem","Democracy","Mixed Dem",1996,1,1
"1590","Taiwan",713,713,"Eastern Asia","Asia","Vincent Siew","Vincent Siew.Taiwan.1997.1999.Mixed Dem","Democracy","Mixed Dem",1997,3,1
"1591","Taiwan",713,713,"Eastern Asia","Asia","Chang Chun-hsiung","Chang Chun-hsiung.Taiwan.2000.2001.Mixed Dem","Democracy","Mixed Dem",2000,2,1
"1592","Taiwan",713,713,"Eastern Asia","Asia","Yu Shyi-kun","Yu Shyi-kun.Taiwan.2002.2004.Mixed Dem","Democracy","Mixed Dem",2002,3,1
"1593","Taiwan",713,713,"Eastern Asia","Asia","Frank Hsieh","Frank Hsieh.Taiwan.2005.2005.Mixed Dem","Democracy","Mixed Dem",2005,1,1
"1594","Taiwan",713,713,"Eastern Asia","Asia","Su Tseng-chang","Su Tseng-chang.Taiwan.2006.2006.Mixed Dem","Democracy","Mixed Dem",2006,1,1
"1595","Taiwan",713,713,"Eastern Asia","Asia","Chang Chun-hsiung","Chang Chun-hsiung.Taiwan.2007.2007.Mixed Dem","Democracy","Mixed Dem",2007,1,1
"1596","Taiwan",713,713,"Eastern Asia","Asia","Liu Chao-shiuan","Liu Chao-shiuan.Taiwan.2008.2008.Mixed Dem","Democracy","Mixed Dem",2008,1,0
"1597","Tajikistan",702,702,"Central Asia","Asia","Rakhmon Nabiyev","Rakhmon Nabiyev.Tajikistan.1991.1991.Civilian Dict","Non-democracy","Civilian Dict",1991,1,1
"1598","Tajikistan",702,702,"Central Asia","Asia","Imomali Rakhmonov","Imomali Rakhmonov.Tajikistan.1992.2008.Civilian Dict","Non-democracy","Civilian Dict",1992,17,0
"1599","Tanzania",510,510,"Eastern Africa","Africa","Julius Nyerere","Julius Nyerere.Tanzania.1961.1984.Civilian Dict","Non-democracy","Civilian Dict",1961,24,1
"1600","Tanzania",510,510,"Eastern Africa","Africa","Ali Hassan Mwinyi","Ali Hassan Mwinyi.Tanzania.1985.1994.Civilian Dict","Non-democracy","Civilian Dict",1985,10,1
"1601","Tanzania",510,510,"Eastern Africa","Africa","Benjamin Mkapa","Benjamin Mkapa.Tanzania.1995.2004.Civilian Dict","Non-democracy","Civilian Dict",1995,10,1
"1602","Tanzania",510,510,"Eastern Africa","Africa","Jakaya Mrisho Kikwete","Jakaya Mrisho Kikwete.Tanzania.2005.2008.Military Dict","Non-democracy","Military Dict",2005,4,0
"1603","Thailand",800,800,"South-Eastern Asia","Asia","Luang Thamrongnawasawat","Luang Thamrongnawasawat.Thailand.1946.1946.Military Dict","Non-democracy","Military Dict",1946,1,1
"1604","Thailand",800,800,"South-Eastern Asia","Asia","Luang Phibulsongkhram","Luang Phibulsongkhram.Thailand.1947.1956.Military Dict","Non-democracy","Military Dict",1947,10,1
"1605","Thailand",800,800,"South-Eastern Asia","Asia","Thanom Kittikachorn","Thanom Kittikachorn.Thailand.1957.1957.Military Dict","Non-democracy","Military Dict",1957,1,1
"1606","Thailand",800,800,"South-Eastern Asia","Asia","Sarit Thanarat","Sarit Thanarat.Thailand.1958.1962.Military Dict","Non-democracy","Military Dict",1958,5,0
"1607","Thailand",800,800,"South-Eastern Asia","Asia","Thanom Kittikachorn","Thanom Kittikachorn.Thailand.1963.1972.Military Dict","Non-democracy","Military Dict",1963,10,1
"1608","Thailand",800,800,"South-Eastern Asia","Asia","Sanya Thammasak","Sanya Thammasak.Thailand.1973.1974.Civilian Dict","Non-democracy","Civilian Dict",1973,2,1
"1609","Thailand",800,800,"South-Eastern Asia","Asia","Kukrit Pramoj","Kukrit Pramoj.Thailand.1975.1975.Parliamentary Dem","Democracy","Parliamentary Dem",1975,1,1
"1610","Thailand",800,800,"South-Eastern Asia","Asia","military","military.Thailand.1976.1976.Military Dict","Non-democracy","Military Dict",1976,1,1
"1611","Thailand",800,800,"South-Eastern Asia","Asia","Kriangsak Chamanand","Kriangsak Chamanand.Thailand.1977.1978.Military Dict","Non-democracy","Military Dict",1977,2,1
"1612","Thailand",800,800,"South-Eastern Asia","Asia","Kriangsak Chamanand","Kriangsak Chamanand.Thailand.1979.1979.Parliamentary Dem","Democracy","Parliamentary Dem",1979,1,1
"1613","Thailand",800,800,"South-Eastern Asia","Asia","Prem Tinsulanond","Prem Tinsulanond.Thailand.1980.1987.Parliamentary Dem","Democracy","Parliamentary Dem",1980,8,1
"1614","Thailand",800,800,"South-Eastern Asia","Asia","Chatichai Choonhavan","Chatichai Choonhavan.Thailand.1988.1990.Parliamentary Dem","Democracy","Parliamentary Dem",1988,3,1
"1615","Thailand",800,800,"South-Eastern Asia","Asia","military","military.Thailand.1991.1991.Military Dict","Non-democracy","Military Dict",1991,1,1
"1616","Thailand",800,800,"South-Eastern Asia","Asia","Chuan Leekpai","Chuan Leekpai.Thailand.1992.1994.Parliamentary Dem","Democracy","Parliamentary Dem",1992,3,1
"1617","Thailand",800,800,"South-Eastern Asia","Asia","Banharn Silapa-archa","Banharn Silapa-archa.Thailand.1995.1995.Parliamentary Dem","Democracy","Parliamentary Dem",1995,1,1
"1618","Thailand",800,800,"South-Eastern Asia","Asia","Chaovalit Yongchaiyudh","Chaovalit Yongchaiyudh.Thailand.1996.1996.Parliamentary Dem","Democracy","Parliamentary Dem",1996,1,1
"1619","Thailand",800,800,"South-Eastern Asia","Asia","Chuan Leekpai","Chuan Leekpai.Thailand.1997.2000.Parliamentary Dem","Democracy","Parliamentary Dem",1997,4,1
"1620","Thailand",800,800,"South-Eastern Asia","Asia","Thaksin Shinawatra","Thaksin Shinawatra.Thailand.2001.2005.Parliamentary Dem","Democracy","Parliamentary Dem",2001,5,1
"1621","Thailand",800,800,"South-Eastern Asia","Asia","Surayud Chulanont","Surayud Chulanont.Thailand.2006.2007.Military Dict","Non-democracy","Military Dict",2006,2,1
"1622","Thailand",800,800,"South-Eastern Asia","Asia","Abhisit Vejjajiva","Abhisit Vejjajiva.Thailand.2008.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2008,1,0
"1623","Togo",461,461,"Western Africa","Africa","Sylvanus Olympio","Sylvanus Olympio.Togo.1960.1962.Civilian Dict","Non-democracy","Civilian Dict",1960,3,0
"1624","Togo",461,461,"Western Africa","Africa","Nicholas Grunitzky","Nicholas Grunitzky.Togo.1963.1966.Civilian Dict","Non-democracy","Civilian Dict",1963,4,1
"1625","Togo",461,461,"Western Africa","Africa","Etienne Eyadema","Etienne Eyadema.Togo.1967.2004.Military Dict","Non-democracy","Military Dict",1967,38,0
"1626","Togo",461,461,"Western Africa","Africa","Faure Gnassingbï¿½","Faure Gnassingbï¿½.Togo.2005.2008.Civilian Dict","Non-democracy","Civilian Dict",2005,4,0
"1627","Tonga",955,955,"Polynesia","Oceania","Taufa'ahau Tupou IV","Taufa'ahau Tupou IV.Tonga.1970.2005.Monarchy","Non-democracy","Monarchy",1970,36,0
"1628","Tonga",955,955,"Polynesia","Oceania","George Tupou V","George Tupou V.Tonga.2006.2008.Monarchy","Non-democracy","Monarchy",2006,3,0
"1629","Trinidad and Tobago",52,52,"Caribbean","Americas","Eric Williams","Eric Williams.Trinidad and Tobago.1962.1980.Parliamentary Dem","Democracy","Parliamentary Dem",1962,19,0
"1630","Trinidad and Tobago",52,52,"Caribbean","Americas","George Chambers","George Chambers.Trinidad and Tobago.1981.1985.Parliamentary Dem","Democracy","Parliamentary Dem",1981,5,1
"1631","Trinidad and Tobago",52,52,"Caribbean","Americas","A.N.R. Robinson","A.N.R. Robinson.Trinidad and Tobago.1986.1990.Parliamentary Dem","Democracy","Parliamentary Dem",1986,5,1
"1632","Trinidad and Tobago",52,52,"Caribbean","Americas","Patrick Manning","Patrick Manning.Trinidad and Tobago.1991.1994.Parliamentary Dem","Democracy","Parliamentary Dem",1991,4,1
"1633","Trinidad and Tobago",52,52,"Caribbean","Americas","Basdeo Panday","Basdeo Panday.Trinidad and Tobago.1995.2000.Parliamentary Dem","Democracy","Parliamentary Dem",1995,6,1
"1634","Trinidad and Tobago",52,52,"Caribbean","Americas","Patrick Manning","Patrick Manning.Trinidad and Tobago.2001.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2001,8,0
"1635","Tunisia",616,616,"Northern Africa","Africa","Habib Bourguiba","Habib Bourguiba.Tunisia.1956.1986.Civilian Dict","Non-democracy","Civilian Dict",1956,31,1
"1636","Tunisia",616,616,"Northern Africa","Africa","Zine al-Abidine Ben Ali","Zine al-Abidine Ben Ali.Tunisia.1987.2008.Military Dict","Non-democracy","Military Dict",1987,22,0
"1637","Turkey",640,640,"Western Asia","Asia","Ismet Inonu","Ismet Inonu.Turkey.1946.1949.Military Dict","Non-democracy","Military Dict",1946,4,1
"1638","Turkey",640,640,"Western Asia","Asia","Celal Bayar","Celal Bayar.Turkey.1950.1959.Civilian Dict","Non-democracy","Civilian Dict",1950,10,1
"1639","Turkey",640,640,"Western Asia","Asia","Cemal Gursel","Cemal Gursel.Turkey.1960.1960.Military Dict","Non-democracy","Military Dict",1960,1,1
"1640","Turkey",640,640,"Western Asia","Asia","Ismet Inonu","Ismet Inonu.Turkey.1961.1964.Parliamentary Dem","Democracy","Parliamentary Dem",1961,4,1
"1641","Turkey",640,640,"Western Asia","Asia","Suleyman Demirel","Suleyman Demirel.Turkey.1965.1970.Parliamentary Dem","Democracy","Parliamentary Dem",1965,6,1
"1642","Turkey",640,640,"Western Asia","Asia","Nihat Erim","Nihat Erim.Turkey.1971.1971.Parliamentary Dem","Democracy","Parliamentary Dem",1971,1,1
"1643","Turkey",640,640,"Western Asia","Asia","Ferit Melen","Ferit Melen.Turkey.1972.1972.Parliamentary Dem","Democracy","Parliamentary Dem",1972,1,1
"1644","Turkey",640,640,"Western Asia","Asia","Naim Talu","Naim Talu.Turkey.1973.1973.Parliamentary Dem","Democracy","Parliamentary Dem",1973,1,1
"1645","Turkey",640,640,"Western Asia","Asia","Said Irmak","Said Irmak.Turkey.1974.1974.Parliamentary Dem","Democracy","Parliamentary Dem",1974,1,1
"1646","Turkey",640,640,"Western Asia","Asia","Suleyman Demirel","Suleyman Demirel.Turkey.1975.1977.Parliamentary Dem","Democracy","Parliamentary Dem",1975,3,1
"1647","Turkey",640,640,"Western Asia","Asia","Bulent Ecevit","Bulent Ecevit.Turkey.1978.1978.Parliamentary Dem","Democracy","Parliamentary Dem",1978,1,1
"1648","Turkey",640,640,"Western Asia","Asia","Suleyman Demirel","Suleyman Demirel.Turkey.1979.1979.Parliamentary Dem","Democracy","Parliamentary Dem",1979,1,1
"1649","Turkey",640,640,"Western Asia","Asia","Kenan Evren","Kenan Evren.Turkey.1980.1982.Military Dict","Non-democracy","Military Dict",1980,3,1
"1650","Turkey",640,640,"Western Asia","Asia","Turgut Ozal","Turgut Ozal.Turkey.1983.1988.Parliamentary Dem","Democracy","Parliamentary Dem",1983,6,1
"1651","Turkey",640,640,"Western Asia","Asia","Yildirim Akbulut","Yildirim Akbulut.Turkey.1989.1990.Parliamentary Dem","Democracy","Parliamentary Dem",1989,2,1
"1652","Turkey",640,640,"Western Asia","Asia","Suleyman Demirel","Suleyman Demirel.Turkey.1991.1992.Parliamentary Dem","Democracy","Parliamentary Dem",1991,2,1
"1653","Turkey",640,640,"Western Asia","Asia","Tansu Ciller","Tansu Ciller.Turkey.1993.1995.Parliamentary Dem","Democracy","Parliamentary Dem",1993,3,1
"1654","Turkey",640,640,"Western Asia","Asia","Necmettin Erbakan","Necmettin Erbakan.Turkey.1996.1996.Parliamentary Dem","Democracy","Parliamentary Dem",1996,1,1
"1655","Turkey",640,640,"Western Asia","Asia","Mesut Yilmaz","Mesut Yilmaz.Turkey.1997.1998.Parliamentary Dem","Democracy","Parliamentary Dem",1997,2,1
"1656","Turkey",640,640,"Western Asia","Asia","Mustafa Bulent Ecevit","Mustafa Bulent Ecevit.Turkey.1999.2001.Parliamentary Dem","Democracy","Parliamentary Dem",1999,3,1
"1657","Turkey",640,640,"Western Asia","Asia","Abdullah Gul","Abdullah Gul.Turkey.2002.2002.Parliamentary Dem","Democracy","Parliamentary Dem",2002,1,1
"1658","Turkey",640,640,"Western Asia","Asia","Recep Tayyip Erdogan","Recep Tayyip Erdogan.Turkey.2003.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2003,6,0
"1659","Turkmenistan",701,701,"Central Asia","Asia","Saparmurad Niyazov","Saparmurad Niyazov.Turkmenistan.1991.2005.Civilian Dict","Non-democracy","Civilian Dict",1991,15,0
"1660","Turkmenistan",701,701,"Central Asia","Asia","Gurbanguly M. Berdymukhammedov","Gurbanguly M. Berdymukhammedov.Turkmenistan.2006.2008.Civilian Dict","Non-democracy","Civilian Dict",2006,3,0
"1661","Tuvalu",947,NA,"Polynesia","Oceania","Toalipi Lauti","Toalipi Lauti.Tuvalu.1978.1980.Parliamentary Dem","Democracy","Parliamentary Dem",1978,3,1
"1662","Tuvalu",947,NA,"Polynesia","Oceania","Tomasi Puapua","Tomasi Puapua.Tuvalu.1981.1988.Parliamentary Dem","Democracy","Parliamentary Dem",1981,8,1
"1663","Tuvalu",947,NA,"Polynesia","Oceania","Bikenibeu Paeniu","Bikenibeu Paeniu.Tuvalu.1989.1992.Parliamentary Dem","Democracy","Parliamentary Dem",1989,4,1
"1664","Tuvalu",947,NA,"Polynesia","Oceania","Kamuta Latasi","Kamuta Latasi.Tuvalu.1993.1995.Parliamentary Dem","Democracy","Parliamentary Dem",1993,3,1
"1665","Tuvalu",947,NA,"Polynesia","Oceania","Bikenibeu Paeniu","Bikenibeu Paeniu.Tuvalu.1996.1998.Parliamentary Dem","Democracy","Parliamentary Dem",1996,3,1
"1666","Tuvalu",947,NA,"Polynesia","Oceania","Ionatana Ionatana","Ionatana Ionatana.Tuvalu.1999.1999.Parliamentary Dem","Democracy","Parliamentary Dem",1999,1,1
"1667","Tuvalu",947,947,"Polynesia","Oceania","Lgitupu Tuilimu","Lgitupu Tuilimu.Tuvalu.2000.2000.Parliamentary Dem","Democracy","Parliamentary Dem",2000,1,1
"1668","Tuvalu",947,947,"Polynesia","Oceania","Koloa Talake","Koloa Talake.Tuvalu.2001.2001.Parliamentary Dem","Democracy","Parliamentary Dem",2001,1,1
"1669","Tuvalu",947,947,"Polynesia","Oceania","Saufatu Sopoanga","Saufatu Sopoanga.Tuvalu.2002.2003.Parliamentary Dem","Democracy","Parliamentary Dem",2002,2,1
"1670","Tuvalu",947,947,"Polynesia","Oceania","Maatia Toafa","Maatia Toafa.Tuvalu.2004.2005.Parliamentary Dem","Democracy","Parliamentary Dem",2004,2,1
"1671","Tuvalu",947,947,"Polynesia","Oceania","Apisai Ielemia","Apisai Ielemia.Tuvalu.2006.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2006,3,0
"1672","U.S.S.R.",365,364,"Eastern Europe","Europe","Josef Stalin","Josef Stalin.U.S.S.R..1946.1952.Civilian Dict","Non-democracy","Civilian Dict",1946,7,0
"1673","U.S.S.R.",365,364,"Eastern Europe","Europe","Nikita Khruschev","Nikita Khruschev.U.S.S.R..1953.1963.Civilian Dict","Non-democracy","Civilian Dict",1953,11,1
"1674","U.S.S.R.",365,364,"Eastern Europe","Europe","Leonid Brezhnev","Leonid Brezhnev.U.S.S.R..1964.1981.Civilian Dict","Non-democracy","Civilian Dict",1964,18,0
"1675","U.S.S.R.",365,364,"Eastern Europe","Europe","Yuri Andropov","Yuri Andropov.U.S.S.R..1982.1983.Civilian Dict","Non-democracy","Civilian Dict",1982,2,0
"1676","U.S.S.R.",365,364,"Eastern Europe","Europe","Konstantin Chernenko","Konstantin Chernenko.U.S.S.R..1984.1984.Civilian Dict","Non-democracy","Civilian Dict",1984,1,0
"1677","U.S.S.R.",365,364,"Eastern Europe","Europe","Mikhail Gorbachev","Mikhail Gorbachev.U.S.S.R..1985.1990.Civilian Dict","Non-democracy","Civilian Dict",1985,6,0
"1678","Uganda",500,500,"Eastern Africa","Africa","Sir Walter Fleming Coutts","Sir Walter Fleming Coutts.Uganda.1962.1962.Civilian Dict","Non-democracy","Civilian Dict",1962,1,1
"1679","Uganda",500,500,"Eastern Africa","Africa","Sir Edward Frederick Mutesa II","Sir Edward Frederick Mutesa II.Uganda.1963.1965.Civilian Dict","Non-democracy","Civilian Dict",1963,3,1
"1680","Uganda",500,500,"Eastern Africa","Africa","Milton Obote","Milton Obote.Uganda.1966.1970.Civilian Dict","Non-democracy","Civilian Dict",1966,5,1
"1681","Uganda",500,500,"Eastern Africa","Africa","Idi Amin Dada","Idi Amin Dada.Uganda.1971.1978.Military Dict","Non-democracy","Military Dict",1971,8,1
"1682","Uganda",500,500,"Eastern Africa","Africa","Godfrey Binsaisa","Godfrey Binsaisa.Uganda.1979.1979.Civilian Dict","Non-democracy","Civilian Dict",1979,1,1
"1683","Uganda",500,500,"Eastern Africa","Africa","Milton Obote","Milton Obote.Uganda.1980.1984.Presidential Dem","Democracy","Presidential Dem",1980,5,1
"1684","Uganda",500,500,"Eastern Africa","Africa","Tito Okello","Tito Okello.Uganda.1985.1985.Military Dict","Non-democracy","Military Dict",1985,1,1
"1685","Uganda",500,500,"Eastern Africa","Africa","Yoweri Museveni","Yoweri Museveni.Uganda.1986.2008.Civilian Dict","Non-democracy","Civilian Dict",1986,23,0
"1686","Ukraine",369,369,"Eastern Europe","Europe","Vitold Pavlovych Fokin","Vitold Pavlovych Fokin.Ukraine.1991.1991.Mixed Dem","Democracy","Mixed Dem",1991,1,1
"1687","Ukraine",369,369,"Eastern Europe","Europe","Leonid Danylovych Kuchma","Leonid Danylovych Kuchma.Ukraine.1992.1992.Mixed Dem","Democracy","Mixed Dem",1992,1,1
"1688","Ukraine",369,369,"Eastern Europe","Europe","Yukhym Leonidovych Zvyahilskiy","Yukhym Leonidovych Zvyahilskiy.Ukraine.1993.1993.Mixed Dem","Democracy","Mixed Dem",1993,1,1
"1689","Ukraine",369,369,"Eastern Europe","Europe","Vitaliy Anriyovych Masol","Vitaliy Anriyovych Masol.Ukraine.1994.1994.Mixed Dem","Democracy","Mixed Dem",1994,1,1
"1690","Ukraine",369,369,"Eastern Europe","Europe","Yevhen Kyrylovych Marchuk","Yevhen Kyrylovych Marchuk.Ukraine.1995.1995.Mixed Dem","Democracy","Mixed Dem",1995,1,1
"1691","Ukraine",369,369,"Eastern Europe","Europe","Pavlo Ivanovych Lazarenko","Pavlo Ivanovych Lazarenko.Ukraine.1996.1996.Mixed Dem","Democracy","Mixed Dem",1996,1,1
"1692","Ukraine",369,369,"Eastern Europe","Europe","Valeriy Pavlovych Pustovoytenko","Valeriy Pavlovych Pustovoytenko.Ukraine.1997.1998.Mixed Dem","Democracy","Mixed Dem",1997,2,1
"1693","Ukraine",369,369,"Eastern Europe","Europe","Viktor Andriyovych Yushchenko","Viktor Andriyovych Yushchenko.Ukraine.1999.2000.Mixed Dem","Democracy","Mixed Dem",1999,2,1
"1694","Ukraine",369,369,"Eastern Europe","Europe","Anatoliy Kyrylovych Kinakh","Anatoliy Kyrylovych Kinakh.Ukraine.2001.2001.Mixed Dem","Democracy","Mixed Dem",2001,1,1
"1695","Ukraine",369,369,"Eastern Europe","Europe","Viktor Fedorovych Yanukovych","Viktor Fedorovych Yanukovych.Ukraine.2002.2004.Mixed Dem","Democracy","Mixed Dem",2002,3,1
"1696","Ukraine",369,369,"Eastern Europe","Europe","Yuriy Ivanovych Yekhanurov","Yuriy Ivanovych Yekhanurov.Ukraine.2005.2005.Mixed Dem","Democracy","Mixed Dem",2005,1,1
"1697","Ukraine",369,369,"Eastern Europe","Europe","Viktor Fedorovych Yanukovych","Viktor Fedorovych Yanukovych.Ukraine.2006.2006.Mixed Dem","Democracy","Mixed Dem",2006,1,1
"1698","Ukraine",369,369,"Eastern Europe","Europe","Yuliya Volodymyrivna Tymoshenko","Yuliya Volodymyrivna Tymoshenko.Ukraine.2007.2008.Mixed Dem","Democracy","Mixed Dem",2007,2,0
"1699","United Arab Emirates",696,696,"Western Asia","Asia","Sheikh Zaid ibn Sultan Al Nahayan","Sheikh Zaid ibn Sultan Al Nahayan.United Arab Emirates.1971.2003.Monarchy","Non-democracy","Monarchy",1971,33,1
"1700","United Arab Emirates",696,696,"Western Asia","Asia","Sheikh Khalifa ibn Zayid Al Nuhayyan","Sheikh Khalifa ibn Zayid Al Nuhayyan.United Arab Emirates.2004.2008.Monarchy","Non-democracy","Monarchy",2004,5,0
"1701","United Kingdom",200,200,"Northern Europe","Europe","Clement Attlee","Clement Attlee.United Kingdom.1946.1950.Parliamentary Dem","Democracy","Parliamentary Dem",1946,5,1
"1702","United Kingdom",200,200,"Northern Europe","Europe","Winston Churchill","Winston Churchill.United Kingdom.1951.1954.Parliamentary Dem","Democracy","Parliamentary Dem",1951,4,1
"1703","United Kingdom",200,200,"Northern Europe","Europe","Anthony Eden","Anthony Eden.United Kingdom.1955.1956.Parliamentary Dem","Democracy","Parliamentary Dem",1955,2,1
"1704","United Kingdom",200,200,"Northern Europe","Europe","Harold MacMillan","Harold MacMillan.United Kingdom.1957.1962.Parliamentary Dem","Democracy","Parliamentary Dem",1957,6,1
"1705","United Kingdom",200,200,"Northern Europe","Europe","Alexander Douglas-Home","Alexander Douglas-Home.United Kingdom.1963.1963.Parliamentary Dem","Democracy","Parliamentary Dem",1963,1,1
"1706","United Kingdom",200,200,"Northern Europe","Europe","Harold Wilson","Harold Wilson.United Kingdom.1964.1969.Parliamentary Dem","Democracy","Parliamentary Dem",1964,6,1
"1707","United Kingdom",200,200,"Northern Europe","Europe","Edward Heath","Edward Heath.United Kingdom.1970.1973.Parliamentary Dem","Democracy","Parliamentary Dem",1970,4,1
"1708","United Kingdom",200,200,"Northern Europe","Europe","Harold Wilson","Harold Wilson.United Kingdom.1974.1975.Parliamentary Dem","Democracy","Parliamentary Dem",1974,2,1
"1709","United Kingdom",200,200,"Northern Europe","Europe","James Callaghan","James Callaghan.United Kingdom.1976.1978.Parliamentary Dem","Democracy","Parliamentary Dem",1976,3,1
"1710","United Kingdom",200,200,"Northern Europe","Europe","Margaret Thatcher","Margaret Thatcher.United Kingdom.1979.1989.Parliamentary Dem","Democracy","Parliamentary Dem",1979,11,1
"1711","United Kingdom",200,200,"Northern Europe","Europe","John Major","John Major.United Kingdom.1990.1996.Parliamentary Dem","Democracy","Parliamentary Dem",1990,7,1
"1712","United Kingdom",200,200,"Northern Europe","Europe","Tony Blair","Tony Blair.United Kingdom.1997.2006.Parliamentary Dem","Democracy","Parliamentary Dem",1997,10,1
"1713","United Kingdom",200,200,"Northern Europe","Europe","Gordon Brown","Gordon Brown.United Kingdom.2007.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2007,2,0
"1714","United States of America",2,2,"Northern America","Americas","Harry Truman","Harry Truman.United States of America.1946.1952.Presidential Dem","Democracy","Presidential Dem",1946,7,1
"1715","United States of America",2,2,"Northern America","Americas","Dwight D. Eisenhower","Dwight D. Eisenhower.United States of America.1953.1960.Presidential Dem","Democracy","Presidential Dem",1953,8,1
"1716","United States of America",2,2,"Northern America","Americas","John Kennedy","John Kennedy.United States of America.1961.1962.Presidential Dem","Democracy","Presidential Dem",1961,2,0
"1717","United States of America",2,2,"Northern America","Americas","Lyndon Johnson","Lyndon Johnson.United States of America.1963.1968.Presidential Dem","Democracy","Presidential Dem",1963,6,1
"1718","United States of America",2,2,"Northern America","Americas","Richard Nixon","Richard Nixon.United States of America.1969.1973.Presidential Dem","Democracy","Presidential Dem",1969,5,1
"1719","United States of America",2,2,"Northern America","Americas","Gerald Ford","Gerald Ford.United States of America.1974.1976.Presidential Dem","Democracy","Presidential Dem",1974,3,1
"1720","United States of America",2,2,"Northern America","Americas","Jimmy Carter","Jimmy Carter.United States of America.1977.1980.Presidential Dem","Democracy","Presidential Dem",1977,4,1
"1721","United States of America",2,2,"Northern America","Americas","Ronald Reagan","Ronald Reagan.United States of America.1981.1988.Presidential Dem","Democracy","Presidential Dem",1981,8,1
"1722","United States of America",2,2,"Northern America","Americas","George Bush","George Bush.United States of America.1989.1992.Presidential Dem","Democracy","Presidential Dem",1989,4,1
"1723","United States of America",2,2,"Northern America","Americas","Bill Clinton","Bill Clinton.United States of America.1993.2000.Presidential Dem","Democracy","Presidential Dem",1993,8,1
"1724","United States of America",2,2,"Northern America","Americas","George W. Bush","George W. Bush.United States of America.2001.2008.Presidential Dem","Democracy","Presidential Dem",2001,8,0
"1725","Uruguay",165,165,"South America","Americas","Juan Amezaga","Juan Amezaga.Uruguay.1946.1946.Presidential Dem","Democracy","Presidential Dem",1946,1,1
"1726","Uruguay",165,165,"South America","Americas","Luis Batlle Berres","Luis Batlle Berres.Uruguay.1947.1950.Presidential Dem","Democracy","Presidential Dem",1947,4,1
"1727","Uruguay",165,165,"South America","Americas","Andres Martinez Trueba","Andres Martinez Trueba.Uruguay.1951.1954.Presidential Dem","Democracy","Presidential Dem",1951,4,1
"1728","Uruguay",165,165,"South America","Americas","Luis Batlle Berres","Luis Batlle Berres.Uruguay.1955.1955.Presidential Dem","Democracy","Presidential Dem",1955,1,1
"1729","Uruguay",165,165,"South America","Americas","Alberto Fermin Zubiria","Alberto Fermin Zubiria.Uruguay.1956.1956.Presidential Dem","Democracy","Presidential Dem",1956,1,1
"1730","Uruguay",165,165,"South America","Americas","Arturo Lezama","Arturo Lezama.Uruguay.1957.1957.Presidential Dem","Democracy","Presidential Dem",1957,1,1
"1731","Uruguay",165,165,"South America","Americas","Carlos Fischer","Carlos Fischer.Uruguay.1958.1958.Presidential Dem","Democracy","Presidential Dem",1958,1,1
"1732","Uruguay",165,165,"South America","Americas","Martin Etchegoyen","Martin Etchegoyen.Uruguay.1959.1959.Presidential Dem","Democracy","Presidential Dem",1959,1,1
"1733","Uruguay",165,165,"South America","Americas","Benito Nardone","Benito Nardone.Uruguay.1960.1960.Presidential Dem","Democracy","Presidential Dem",1960,1,1
"1734","Uruguay",165,165,"South America","Americas","Eduardo Haedo","Eduardo Haedo.Uruguay.1961.1961.Presidential Dem","Democracy","Presidential Dem",1961,1,1
"1735","Uruguay",165,165,"South America","Americas","Faustino Harrison","Faustino Harrison.Uruguay.1962.1962.Presidential Dem","Democracy","Presidential Dem",1962,1,1
"1736","Uruguay",165,165,"South America","Americas","Daniel Fernandez Crespo","Daniel Fernandez Crespo.Uruguay.1963.1963.Presidential Dem","Democracy","Presidential Dem",1963,1,1
"1737","Uruguay",165,165,"South America","Americas","Luis Giannatasio","Luis Giannatasio.Uruguay.1964.1964.Presidential Dem","Democracy","Presidential Dem",1964,1,1
"1738","Uruguay",165,165,"South America","Americas","Washington Beltran","Washington Beltran.Uruguay.1965.1965.Presidential Dem","Democracy","Presidential Dem",1965,1,1
"1739","Uruguay",165,165,"South America","Americas","Alberto Heber Usher","Alberto Heber Usher.Uruguay.1966.1966.Presidential Dem","Democracy","Presidential Dem",1966,1,1
"1740","Uruguay",165,165,"South America","Americas","Jorge Pacheco Arego","Jorge Pacheco Arego.Uruguay.1967.1971.Presidential Dem","Democracy","Presidential Dem",1967,5,1
"1741","Uruguay",165,165,"South America","Americas","Juan Bordaberry Arocena","Juan Bordaberry Arocena.Uruguay.1972.1972.Presidential Dem","Democracy","Presidential Dem",1972,1,1
"1742","Uruguay",165,165,"South America","Americas","Military","Military.Uruguay.1973.1980.Military Dict","Non-democracy","Military Dict",1973,8,1
"1743","Uruguay",165,165,"South America","Americas","Gregorio Alvarez Armelino","Gregorio Alvarez Armelino.Uruguay.1981.1984.Military Dict","Non-democracy","Military Dict",1981,4,1
"1744","Uruguay",165,165,"South America","Americas","Julio Sanguinetti Cairdo","Julio Sanguinetti Cairdo.Uruguay.1985.1989.Presidential Dem","Democracy","Presidential Dem",1985,5,1
"1745","Uruguay",165,165,"South America","Americas","Luis Alberto Lacalle","Luis Alberto Lacalle.Uruguay.1990.1994.Presidential Dem","Democracy","Presidential Dem",1990,5,1
"1746","Uruguay",165,165,"South America","Americas","Julio Sanguinetti Cairdo","Julio Sanguinetti Cairdo.Uruguay.1995.1999.Presidential Dem","Democracy","Presidential Dem",1995,5,1
"1747","Uruguay",165,165,"South America","Americas","Jorge Luis Batlle Ibanez","Jorge Luis Batlle Ibanez.Uruguay.2000.2004.Presidential Dem","Democracy","Presidential Dem",2000,5,1
"1748","Uruguay",165,165,"South America","Americas","Tabarï¿½ Ramï¿½n Vï¿½zquez Rosas","Tabarï¿½ Ramï¿½n Vï¿½zquez Rosas.Uruguay.2005.2008.Presidential Dem","Democracy","Presidential Dem",2005,4,0
"1749","Uzbekistan",704,704,"Central Asia","Asia","Islam Karimov","Islam Karimov.Uzbekistan.1991.2008.Civilian Dict","Non-democracy","Civilian Dict",1991,18,0
"1750","Vanuatu",935,935,"Melanesia","Oceania","Walter Lini","Walter Lini.Vanuatu.1980.1990.Parliamentary Dem","Democracy","Parliamentary Dem",1980,11,1
"1751","Vanuatu",935,935,"Melanesia","Oceania","Maxime Carlot Korman","Maxime Carlot Korman.Vanuatu.1991.1994.Parliamentary Dem","Democracy","Parliamentary Dem",1991,4,1
"1752","Vanuatu",935,935,"Melanesia","Oceania","Serge Vohor","Serge Vohor.Vanuatu.1995.1997.Parliamentary Dem","Democracy","Parliamentary Dem",1995,3,1
"1753","Vanuatu",935,935,"Melanesia","Oceania","Donald Kalpokas","Donald Kalpokas.Vanuatu.1998.1998.Parliamentary Dem","Democracy","Parliamentary Dem",1998,1,1
"1754","Vanuatu",935,935,"Melanesia","Oceania","Barak Sope","Barak Sope.Vanuatu.1999.2000.Parliamentary Dem","Democracy","Parliamentary Dem",1999,2,1
"1755","Vanuatu",935,935,"Melanesia","Oceania","Edward Natapei","Edward Natapei.Vanuatu.2001.2003.Parliamentary Dem","Democracy","Parliamentary Dem",2001,3,1
"1756","Vanuatu",935,935,"Melanesia","Oceania","Ham Lini","Ham Lini.Vanuatu.2004.2007.Parliamentary Dem","Democracy","Parliamentary Dem",2004,4,1
"1757","Vanuatu",935,935,"Melanesia","Oceania","Edward Natapei","Edward Natapei.Vanuatu.2008.2008.Parliamentary Dem","Democracy","Parliamentary Dem",2008,1,0
"1758","Venezuela",101,101,"South America","Americas","Romulo Betancourt","Romulo Betancourt.Venezuela.1946.1947.Presidential Dem","Democracy","Presidential Dem",1946,2,1
"1759","Venezuela",101,101,"South America","Americas","Carlos Delgado Chalbaud","Carlos Delgado Chalbaud.Venezuela.1948.1949.Military Dict","Non-democracy","Military Dict",1948,2,0
"1760","Venezuela",101,101,"South America","Americas","Marcos Perez Jimenez","Marcos Perez Jimenez.Venezuela.1950.1957.Military Dict","Non-democracy","Military Dict",1950,8,1
"1761","Venezuela",101,101,"South America","Americas","military","military.Venezuela.1958.1958.Military Dict","Non-democracy","Military Dict",1958,1,1
"1762","Venezuela",101,101,"South America","Americas","Romulo Betancourt","Romulo Betancourt.Venezuela.1959.1963.Presidential Dem","Democracy","Presidential Dem",1959,5,1
"1763","Venezuela",101,101,"South America","Americas","Raul Leoni","Raul Leoni.Venezuela.1964.1968.Presidential Dem","Democracy","Presidential Dem",1964,5,1
"1764","Venezuela",101,101,"South America","Americas","Rafael Caldera Rodriguez","Rafael Caldera Rodriguez.Venezuela.1969.1973.Presidential Dem","Democracy","Presidential Dem",1969,5,1
"1765","Venezuela",101,101,"South America","Americas","Carlos Andres Perez","Carlos Andres Perez.Venezuela.1974.1978.Presidential Dem","Democracy","Presidential Dem",1974,5,1
"1766","Venezuela",101,101,"South America","Americas","Luis Herrera Campins","Luis Herrera Campins.Venezuela.1979.1983.Presidential Dem","Democracy","Presidential Dem",1979,5,1
"1767","Venezuela",101,101,"South America","Americas","Jaime Lusinchi","Jaime Lusinchi.Venezuela.1984.1988.Presidential Dem","Democracy","Presidential Dem",1984,5,1
"1768","Venezuela",101,101,"South America","Americas","Carlos Andres Perez","Carlos Andres Perez.Venezuela.1989.1992.Presidential Dem","Democracy","Presidential Dem",1989,4,1
"1769","Venezuela",101,101,"South America","Americas","Ramon Jose Velasquez","Ramon Jose Velasquez.Venezuela.1993.1993.Presidential Dem","Democracy","Presidential Dem",1993,1,1
"1770","Venezuela",101,101,"South America","Americas","Rafael Caldera Rodriguez","Rafael Caldera Rodriguez.Venezuela.1994.1998.Presidential Dem","Democracy","Presidential Dem",1994,5,1
"1771","Venezuela",101,101,"South America","Americas","Hugo Chavez","Hugo Chavez.Venezuela.1999.2008.Presidential Dem","Democracy","Presidential Dem",1999,10,0
"1772","Viet Nam",818,818,"South-Eastern Asia","Asia","Le Duan","Le Duan.Viet Nam.1976.1985.Civilian Dict","Non-democracy","Civilian Dict",1976,10,0
"1773","Viet Nam",818,818,"South-Eastern Asia","Asia","Nguyen Van Linh","Nguyen Van Linh.Viet Nam.1986.1990.Civilian Dict","Non-democracy","Civilian Dict",1986,5,1
"1774","Viet Nam",818,818,"South-Eastern Asia","Asia","Do Muoi","Do Muoi.Viet Nam.1991.1996.Civilian Dict","Non-democracy","Civilian Dict",1991,6,1
"1775","Viet Nam",818,818,"South-Eastern Asia","Asia","Le Kha Phieu","Le Kha Phieu.Viet Nam.1997.2000.Military Dict","Non-democracy","Military Dict",1997,4,1
"1776","Viet Nam",818,818,"South-Eastern Asia","Asia","Nong Duc Manh","Nong Duc Manh.Viet Nam.2001.2008.Civilian Dict","Non-democracy","Civilian Dict",2001,8,0
"1777","Yemen",679,679,"Western Asia","Asia","Ali Abdullah Saleh","Ali Abdullah Saleh.Yemen.1990.2008.Military Dict","Non-democracy","Military Dict",1990,19,0
"1778","Yemen Arab Republic",678,678,"Western Asia","Asia","Yahya ibn Muhammad","Yahya ibn Muhammad.Yemen Arab Republic.1946.1947.Monarchy","Non-democracy","Monarchy",1946,2,0
"1779","Yemen Arab Republic",678,678,"Western Asia","Asia","Ahmad ash-Shams ibn Yahya","Ahmad ash-Shams ibn Yahya.Yemen Arab Republic.1948.1961.Monarchy","Non-democracy","Monarchy",1948,14,0
"1780","Yemen Arab Republic",678,678,"Western Asia","Asia","`Abd Allah as-Sallal","`Abd Allah as-Sallal.Yemen Arab Republic.1962.1966.Civilian Dict","Non-democracy","Civilian Dict",1962,5,1
"1781","Yemen Arab Republic",678,678,"Western Asia","Asia","Abdel Rahman al-Iriani","Abdel Rahman al-Iriani.Yemen Arab Republic.1967.1973.Civilian Dict","Non-democracy","Civilian Dict",1967,7,1
"1782","Yemen Arab Republic",678,678,"Western Asia","Asia","Ibrahim Muhammad al-Hamadi","Ibrahim Muhammad al-Hamadi.Yemen Arab Republic.1974.1976.Military Dict","Non-democracy","Military Dict",1974,3,0
"1783","Yemen Arab Republic",678,678,"Western Asia","Asia","Ahmad al-Ghashmi","Ahmad al-Ghashmi.Yemen Arab Republic.1977.1977.Military Dict","Non-democracy","Military Dict",1977,1,0
"1784","Yemen Arab Republic",678,678,"Western Asia","Asia","Ali Abdullah Saleh","Ali Abdullah Saleh.Yemen Arab Republic.1978.1989.Military Dict","Non-democracy","Military Dict",1978,12,0
"1785","Yemen PDR (South)",680,680,"Western Asia","Asia","Qahtan Muhammad al-Shabi","Qahtan Muhammad al-Shabi.Yemen PDR (South).1967.1968.Civilian Dict","Non-democracy","Civilian Dict",1967,2,1
"1786","Yemen PDR (South)",680,680,"Western Asia","Asia","Salem Ali Rubayyi","Salem Ali Rubayyi.Yemen PDR (South).1969.1977.Civilian Dict","Non-democracy","Civilian Dict",1969,9,0
"1787","Yemen PDR (South)",680,680,"Western Asia","Asia","Abd al-Fattah Ismail","Abd al-Fattah Ismail.Yemen PDR (South).1978.1979.Civilian Dict","Non-democracy","Civilian Dict",1978,2,1
"1788","Yemen PDR (South)",680,680,"Western Asia","Asia","Ali Nasir Muhammad Husani","Ali Nasir Muhammad Husani.Yemen PDR (South).1980.1985.Civilian Dict","Non-democracy","Civilian Dict",1980,6,1
"1789","Yemen PDR (South)",680,680,"Western Asia","Asia","Haidar Abu Bakr al-Attas","Haidar Abu Bakr al-Attas.Yemen PDR (South).1986.1989.Civilian Dict","Non-democracy","Civilian Dict",1986,4,0
"1790","Yugoslavia",345,345,"Southern Europe","Europe","Josip Broz Tito","Josip Broz Tito.Yugoslavia.1946.1979.Civilian Dict","Non-democracy","Civilian Dict",1946,34,0
"1791","Yugoslavia",345,345,"Southern Europe","Europe","Lazar Mojsov","Lazar Mojsov.Yugoslavia.1980.1980.Civilian Dict","Non-democracy","Civilian Dict",1980,1,1
"1792","Yugoslavia",345,345,"Southern Europe","Europe","Dusan Dragosavac","Dusan Dragosavac.Yugoslavia.1981.1981.Civilian Dict","Non-democracy","Civilian Dict",1981,1,1
"1793","Yugoslavia",345,345,"Southern Europe","Europe","Mitja Ribicic","Mitja Ribicic.Yugoslavia.1982.1982.Civilian Dict","Non-democracy","Civilian Dict",1982,1,1
"1794","Yugoslavia",345,345,"Southern Europe","Europe","Dragoslav Markovic","Dragoslav Markovic.Yugoslavia.1983.1983.Civilian Dict","Non-democracy","Civilian Dict",1983,1,1
"1795","Yugoslavia",345,345,"Southern Europe","Europe","Ali Sukrija","Ali Sukrija.Yugoslavia.1984.1984.Civilian Dict","Non-democracy","Civilian Dict",1984,1,1
"1796","Yugoslavia",345,345,"Southern Europe","Europe","Vidoje Zarkovic","Vidoje Zarkovic.Yugoslavia.1985.1985.Civilian Dict","Non-democracy","Civilian Dict",1985,1,1
"1797","Yugoslavia",345,345,"Southern Europe","Europe","Milanko Renovica","Milanko Renovica.Yugoslavia.1986.1986.Civilian Dict","Non-democracy","Civilian Dict",1986,1,1
"1798","Yugoslavia",345,345,"Southern Europe","Europe","Bosko Krunic","Bosko Krunic.Yugoslavia.1987.1987.Civilian Dict","Non-democracy","Civilian Dict",1987,1,1
"1799","Yugoslavia",345,345,"Southern Europe","Europe","Stipe Suvar","Stipe Suvar.Yugoslavia.1988.1988.Civilian Dict","Non-democracy","Civilian Dict",1988,1,1
"1800","Yugoslavia",345,345,"Southern Europe","Europe","Milan Pancevski","Milan Pancevski.Yugoslavia.1989.1989.Civilian Dict","Non-democracy","Civilian Dict",1989,1,1
"1801","Yugoslavia",345,345,"Southern Europe","Europe","Borisav Jovic","Borisav Jovic.Yugoslavia.1990.1990.Civilian Dict","Non-democracy","Civilian Dict",1990,1,0
"1802","Zambia",551,551,"Eastern Africa","Africa","Kenneth Kaunda","Kenneth Kaunda.Zambia.1964.1990.Civilian Dict","Non-democracy","Civilian Dict",1964,27,1
"1803","Zambia",551,551,"Eastern Africa","Africa","Frederick Chiluba","Frederick Chiluba.Zambia.1991.2001.Civilian Dict","Non-democracy","Civilian Dict",1991,11,1
"1804","Zambia",551,551,"Eastern Africa","Africa","Levy Patrick Mwanawasa","Levy Patrick Mwanawasa.Zambia.2002.2007.Civilian Dict","Non-democracy","Civilian Dict",2002,6,1
"1805","Zambia",551,551,"Eastern Africa","Africa","Rupiah Bwezani Banda","Rupiah Bwezani Banda.Zambia.2008.2008.Civilian Dict","Non-democracy","Civilian Dict",2008,1,0
"1806","Zimbabwe",552,552,"Eastern Africa","Africa","Ian Smith","Ian Smith.Zimbabwe.1965.1978.Civilian Dict","Non-democracy","Civilian Dict",1965,14,1
"1807","Zimbabwe",552,552,"Eastern Africa","Africa","Abel Muzorewa","Abel Muzorewa.Zimbabwe.1979.1979.Civilian Dict","Non-democracy","Civilian Dict",1979,1,1
"1808","Zimbabwe",552,552,"Eastern Africa","Africa","Robert Mugabe","Robert Mugabe.Zimbabwe.1980.2008.Civilian Dict","Non-democracy","Civilian Dict",1980,29,0"""),
index_col=[0], header=1)


def generate_regression_dataset():
    return pd.read_csv(
    StringIO(u"""
var1,var2,var3,var4,T,E
0.5951697438546901,1.14347169375119,1.5710788740267934,1.0,14.785478547854785,1
0.20932541672176813,0.18467669526521932,0.356980285734657,1.0,7.336733673367337,1
0.6939189256884803,0.07189290564119132,0.5579596369568557,1.0,5.271527152715271,1
0.44380363548686375,1.3646463260393265,0.3742214001155121,1.0,11.684168416841684,1
1.6133244141559746,0.12556628933069874,1.9213247307606491,1.0,7.637763776377637,1
0.06563624239055911,0.09837537607740067,0.23789556103788456,1.0,12.678267826782678,1
0.3862944076628313,1.6630915467299439,0.7903142211383773,1.0,6.601660166016601,1
0.9466875753512356,1.3453940656312007,3.20911268157136,1.0,11.369136913691369,1
0.11373961635331922,0.40986001192916977,0.06493423455918193,1.0,14.68046804680468,1
0.7777928381300385,0.33498991375193715,0.41105454455708185,1.0,10.585058505850585,1
0.04428031892216578,0.3051578045560708,0.17648044072321029,1.0,19.37093709370937,1
1.035449993175163,3.3047331895411323,0.9973229240273257,1.0,5.558555855585558,1
0.22919500985314345,0.5813553439220956,0.4847935384909877,1.0,11.292129212921292,1
0.055970031448070424,2.6741349607106155,0.355278796648057,1.0,9.91999199919992,0
1.2365825459439084,1.7965983662360094,0.1799518612021181,1.0,9.884988498849884,1
1.162834828839744,0.4647589263047849,2.028853965185938,1.0,6.265626562656266,1
0.14943646240437114,2.949290606109733,0.2778013526657975,1.0,13.812381238123812,1
0.39947496343506655,0.8224126950445942,0.6734045646168838,1.0,6.433643364336434,1
0.7621212350351054,0.05040707199185315,1.2851628416406744,1.0,6.979697969796979,1
1.2397184972378472,1.8692145634486852,0.020202165664896698,1.0,7.742774277427743,1
0.01922067132173192,1.4355432694660155,0.2556889900950628,1.0,4.704470447044704,1
0.09025297903613075,0.21103684560087815,0.37280945055673376,1.0,11.236123612361236,1
0.20584865589788634,0.04872152939830475,0.002529886542020026,1.0,6.664666466646665,1
0.08818541145883692,1.319678843244738,0.20167456926415986,1.0,10.718071807180717,1
4.629747231213237,0.3635220036804111,1.082069652236349,1.0,11.593159315931592,1
1.6028357835065097,1.2178806764186096,0.35083696741050086,1.0,8.463846384638464,1
0.014803824363587861,0.684736603307301,0.49326684311601293,1.0,5.432543254325433,1
0.08402018524843372,1.4320927559100234,0.45654106320201726,1.0,10.277027702770276,1
2.2602231152333183,1.2389295766737747,5.541837905521401,1.0,7.217721772177217,1
0.6219675941549933,0.6844282332663192,0.13593343710135816,1.0,13.217321732173216,1
0.013219618434182224,3.280555158901928,1.1935510257771664,1.0,8.02980298029803,1
0.07065061567146355,1.4305166496493416,0.005205103509701803,1.0,9.807980798079807,1
0.20598001665787236,0.2906403268309732,0.09656491489599246,1.0,2.632263226322632,1
1.3898823754015133,0.14313310089044293,0.8212569616609695,1.0,6.16061606160616,1
0.1041428509032402,2.0729237697158687,0.44969632462401904,1.0,6.265626562656266,1
0.42848094029362643,0.0657386549672956,3.007549641542905,1.0,4.6064606460646065,0
1.7851514646797528,1.572281968110713,0.4750592215655808,1.0,11.124112411241123,1
0.2288930857061604,0.4290248263765111,0.6080495332734298,1.0,6.083608360836084,1
0.6408374226724668,0.3110844239838485,3.165658013979253,1.0,9.21992199219922,1
1.4506834585050246,0.8470215479051397,2.5211773096233125,1.0,7.11971197119712,0
0.4694585306561667,0.3188706619292891,0.1644982392688507,1.0,13.056305630563056,1
0.36617032773460906,0.23499054954373066,0.6787089172155912,1.0,4.949494949494949,1
1.4325700749632688,2.6683348373975835,0.5580459839005906,1.0,4.858485848584858,1
2.6964625114706475,0.2440771927392136,1.3151113425711525,1.0,8.505850585058505,1
0.1526241013743321,0.3795012266932666,0.33016412502250847,1.0,7.035703570357035,1
0.2773992396102305,0.8716028944125993,1.5551848689645897,1.0,10.837083708370837,1
0.35363295058305305,0.2942364717604685,0.928572980260901,1.0,9.91999199919992,1
0.6209559765590129,0.021883945410842317,3.2057599432524446,1.0,5.292529252925292,1
0.0007568304763381865,1.2166148822226333,0.861069144884359,1.0,20.98109810981098,1
0.49767424081959755,2.7440324329670998,0.47358881276915743,1.0,5.810581058105811,1
1.2137088085905934,0.07275599207614739,0.09841971036529058,1.0,11.292129212921292,1
0.42625505542983516,2.5503918593511834,0.16762004376155556,1.0,8.197819781978197,1
0.008407884071948186,1.1322046751704602,1.234917068207736,1.0,7.469746974697469,1
1.2078327963731073,0.13335024539249735,0.5282312740233687,1.0,10.669066906690668,1
0.03697513660742381,0.04063127750698724,0.26639991908008803,1.0,10.543054305430543,1
0.7894393169081668,0.6690671614657788,1.332696669959017,1.0,6.286628662866287,1
1.4820545200203379,0.6272053107704317,0.7382709520263689,1.0,9.07990799079908,1
1.0286706763487006,0.21520971284363008,0.4576922032422134,1.0,14.183418341834184,1
0.5219860974428132,2.2826832385597053,0.3159760374764989,1.0,21.94019401940194,1
0.26262150375034216,0.34599897459539214,0.9210972350271455,1.0,8.883888388838884,1
0.3603185292736921,1.0013639154752498,0.23753343016870013,1.0,9.982998299829983,1
0.36258730978535225,0.11004608090501279,2.486690734049395,1.0,9.555955595559556,1
1.793597848573506,0.31000116451013165,0.2630657683775386,1.0,8.65986598659866,1
0.41927486658740915,0.11430798994374779,1.1247838155390644,1.0,5.642564256425643,1
1.4702472205197898,0.2890540418325013,0.3318326887742901,1.0,10.998099809980998,1
0.27476015833462675,0.5235084940908867,2.1392038030338645,1.0,8.05080508050805,1
0.11980513172176577,0.7337739574393896,0.21205713898221487,1.0,11.25012501250125,1
0.36929383231374174,0.6098469611582363,0.8940195386288149,1.0,10.214021402140213,0
1.0182500552417189,2.11966580239176,0.7160022115370389,1.0,12.335233523352334,1
0.6070648773862332,2.35011157373498,0.031389772541161494,1.0,15.723572357235723,1
4.169830595980441,0.316285150306315,0.16934997037199687,1.0,11.83118311831183,0
1.4833831655715266,2.2427435143701286,0.2654297513609948,1.0,7.364736473647365,1
0.3235970939468162,0.16515876334551072,0.972039635805659,1.0,12.517251725172517,1
1.8271599686965596,0.32779377433995044,0.9415389427692891,1.0,8.512851285128512,1
0.1041042817799716,0.9233022552016643,1.220070449306738,1.0,12.797279727972796,1
0.39276556698481113,0.42279366106549976,3.4826081972861176,1.0,8.54085408540854,1
2.5796288001231322,0.10901056442071824,2.2800280401047317,1.0,3.5633563356335634,1
0.7750915163023366,0.9745192277909625,2.223698799162753,1.0,9.576957695769577,1
0.5075933616804139,0.9172783608153234,0.103131049110046,1.0,9.646964696469647,1
0.13843715360247075,2.4740843287186447,1.6350046807164011,1.0,11.789178917891789,1
1.120585964966159,1.4805929626236445,0.6382436960512384,1.0,4.648464846484648,0
0.0018418064186871458,0.601451742117251,0.40550997208118705,1.0,14.162416241624163,1
0.996979183052272,0.44858987576726395,0.7820131464779746,1.0,7.490749074907491,1
1.0672627247452178,0.3045821497186961,0.7952756079285181,1.0,10.739073907390738,1
1.123429379093065,1.4093153741348754,0.09089532800894148,1.0,8.23982398239824,1
0.13915819978553629,3.2035226008803197,0.2873492727969337,1.0,7.63076307630763,1
1.2765289009959737,1.0393129471235372,1.217826939298198,1.0,7.8197819781978195,1
0.17582373255135528,1.3716354861239075,1.7854883253168858,1.0,12.419241924192418,1
0.2413014755842905,4.0488064818171585,0.4234150958535507,1.0,10.564056405640564,1
1.8644438653539037,0.821838850359624,0.42636362090628127,1.0,6.293629362936294,1
0.3402951177961094,0.727142869677613,0.3414365394082265,1.0,12.405240524052404,1
5.130831208190108,0.07451281215759868,0.8015256927889071,1.0,10.795079507950796,1
1.4046349355543402,0.0392514176281585,0.7851619729399375,1.0,17.095709570957094,1
0.07394369314978207,0.05331475244292909,0.18626177356664297,1.0,15.464546454645465,0
1.2714884844687595,0.10678035319395672,0.29188268906478476,1.0,9.611961196119612,1
0.7814515900326462,1.2290763370760032,0.0697470619920518,1.0,14.407440744074407,1
0.39089997289660666,0.35690703998590806,0.23057954853260001,1.0,10.088008800880088,1
2.1938248595055168,0.621183576172037,0.466924902554038,1.0,5.817581758175818,1
2.9428820120534365,0.16382954233090405,1.04033340264097,1.0,7.987798779877988,1
0.7055272432734351,0.5926991549723539,0.9232476607991905,1.0,11.83118311831183,1
1.6629246762353496,2.1851704442088487,0.6642734744435368,1.0,11.873187318731873,1
0.4078422870872072,1.0116114336001172,0.4855915992362707,1.0,4.144414441444145,1
0.09132052333724856,0.2815928425136233,0.15394689976472442,1.0,8.28882888288829,1
3.538508770258959,1.807149547414044,1.3369613781592369,1.0,4.326432643264327,1
0.6610269824523992,1.171562626007443,0.3009103939819197,1.0,14.246424642464246,1
0.1065518826286166,0.12184291458381893,0.25787799749473406,1.0,5.6635663566356635,1
0.10432734740269538,1.5135027188196712,0.3145811973140573,1.0,7.917791779177918,1
0.81183737126959,1.683324264378186,0.061924630079522834,1.0,11.845184518451845,1
0.40249470639559076,0.43152022521956473,0.489576068615187,1.0,11.075107510751074,1
1.3221547483209852,0.5211605610933655,1.8599885159772198,1.0,6.888688868886889,1
0.6479539777730214,3.243630978125018,0.03407509189635303,1.0,8.344834483448345,1
0.8514762575346934,0.21736574780851486,0.29732984457567796,1.0,4.473447344734473,1
0.14999350767872174,3.0278894414152338,0.5427493733594075,1.0,9.247924792479248,1
0.38127563617891547,1.1469272170647982,0.22582995205701278,1.0,11.215121512151216,1
0.019479396520878958,1.3747073655938449,1.566595370493247,1.0,8.28882888288829,1
0.8067931024376075,0.6094099818980567,1.9036484933942928,1.0,10.074007400740074,1
0.9268283132677622,1.062157952264051,0.048544293411045755,1.0,14.484448444844483,1
0.9982817051401587,0.3859108590965986,1.4033046561457174,1.0,11.026102610261026,1
0.19875515757596698,1.6686754961119115,0.18233677832237064,1.0,6.251625162516252,1
1.6682316310403287,0.7171129015561293,0.3931803472896087,1.0,15.884588458845885,1
0.9033880759262374,0.347570437567475,0.7962154883808824,1.0,11.341134113411341,1
3.0942165873660246,0.764496697566433,3.063756167185769,1.0,7.644764476447644,1
0.5657649074699271,0.8556440187363783,2.4122224772037586,1.0,8.365836583658366,1
0.6005436656405767,0.019665778638387768,2.356106627011514,1.0,11.9011901190119,1
0.4532005593394602,0.24214916088874208,0.7611142068290219,1.0,9.912991299129914,1
0.44160471133441986,0.2713662930202241,0.9775216177363376,1.0,8.323832383238324,1
0.4113495355835634,0.02948423420287134,1.8434576889665502,1.0,11.97119711971197,1
0.5351195112288876,0.045628976816798617,0.16006748628900558,1.0,11.124112411241123,1
0.47211868864865836,2.239748581628323,0.1488282915995624,1.0,6.153615361536153,1
0.48575359774946963,1.4640131897629343,0.3802925034631497,1.0,8.911891189118911,1
5.353937453111672,0.8552983638136195,0.001005773237084357,1.0,4.879487948794879,1
0.000973612715396856,0.35496485644483466,0.6987414240173688,1.0,20.666066606660667,1
0.36145744710685124,2.7928620225050236,1.503787197764641,1.0,11.082108210821081,1
1.2026725129318079,1.8258521262972167,0.3913388032603664,1.0,8.008800880088009,1
0.8530773930337098,0.22137649239185922,1.6355386168917376,1.0,9.77997799779978,1
1.6469585179126762,3.3371691064843714,1.2626716544855696,1.0,5.6635663566356635,1
0.05049054902113934,1.0423882443442882,0.04040618497708111,1.0,10.501050105010501,1
0.6930326688921491,0.06771662251988092,1.6319301984004901,1.0,9.96899689968997,1
3.8753611845656892,1.206578736767013,0.6567852627734115,1.0,4.837483748374837,1
0.40175441214072477,1.5264428121925269,0.4496207580049677,1.0,7.952795279527953,1
2.1121413204322472,0.9946038239108969,0.1259283018587393,1.0,4.445444544454445,1
2.358111107090637,1.4111736249387716,4.747023248329813,1.0,6.930693069306931,1
0.40616651352273575,0.7479362433150686,1.2402329803968728,1.0,11.97119711971197,1
0.9833121137092842,1.3306985212515354,0.9310574260170132,1.0,12.769276927692768,1
2.802856370352899,0.14176777888646142,0.9644696142808675,1.0,6.153615361536153,1
0.22597951055292706,0.1569690586253124,0.7716777832341211,1.0,9.093909390939094,1
1.0202115825257667,1.338746533401938,1.4854066003255249,1.0,9.702970297029703,1
0.7371829653786988,0.2119671350155468,1.479703406356688,1.0,10.417041704170417,1
0.6945961478381529,0.1330654844727483,1.6121992246696701,1.0,13.182318231823182,1
1.614919107081387,1.6284138939761872,3.339563015085536,1.0,2.576257625762576,1
1.2635668936294953,0.04162647624113071,0.1344880264732324,1.0,7.091709170917092,1
1.8175903742399115,0.893709731112723,0.25683098098044016,1.0,5.74057405740574,1
0.22144152415449195,1.0004704921949568,0.13556537030145238,1.0,12.923292329232924,1
0.3885705443767488,2.3313119236097024,0.04811740463245047,1.0,12.874287428742875,1
1.3654614400117282,0.4447297819814065,0.26387964117279616,1.0,4.725472547254725,1
0.01744557059982297,1.5025104319488611,1.8596484061179372,1.0,9.835983598359835,1
0.8032167120304585,0.2596778548269076,0.30569547402326314,1.0,6.062606260626063,1
1.153737630194303,2.357564941097182,0.26492479564696175,1.0,8.092809280928092,1
0.5464251751430853,0.5165248611390022,0.059806060097181105,1.0,8.043804380438043,1
0.06136716316949647,2.453070810845582,0.23481646918124058,1.0,8.715871587158716,1
0.4211356453163569,0.295454652049226,1.1176635745546923,1.0,13.287328732873288,1
1.5747787981367054,0.7411221082974356,0.5336759550703262,1.0,10.515051505150515,1
1.3943506727733175,0.8777931273877497,1.6376518600382237,1.0,6.426642664266426,1
0.9234411598154695,1.1076137130295087,0.7829099206061244,1.0,3.6403640364036405,1
0.23134585982478487,0.6201351466524622,1.8213552975523695,1.0,4.746474647464747,1
0.735705618321913,3.405054070753984,3.4576251010339982,1.0,11.677167716771677,1
1.748839436422709,1.1326281013809654,0.812583892640099,1.0,11.558155815581557,1
0.2802913861001123,1.6648371474881212,0.051459818339006314,1.0,8.757875787578758,1
0.15085666302889103,2.545695939299097,1.4561193695377936,1.0,12.825282528252824,1
1.5516807175277625,0.12511438017073526,0.14835546399237348,1.0,15.618561856185618,0
0.7463879737526906,0.2674582954863265,0.4200361011988838,1.0,11.943194319431942,1
0.06817704542237235,0.19378821634023047,2.693532634591712,1.0,7.952795279527953,1
0.3051408759563596,0.8589878738431004,3.883753409553651,1.0,12.356235623562355,1
3.6149558299876854,0.6597836973736433,1.013163923813801,1.0,3.5633563356335634,1
1.9810328319378134,0.737971996441928,0.2720711604679828,1.0,8.561856185618561,1
0.19708044679622314,1.1649575645233563,0.8204869845982585,1.0,4.207420742074207,1
0.027854134406850403,0.6533255821466247,0.08022010668090788,1.0,21.03010301030103,1
1.8066664037191318,3.535072020442403,2.1767587869903036,1.0,5.810581058105811,1
0.16528780485551056,1.6233946294015358,1.994550831122152,1.0,8.7998799879988,1
1.6170625036979323,0.4947975295247246,0.1315965858883526,1.0,7.798779877987799,0
1.2987944075293532,1.7780364608174435,0.4536930070119434,1.0,12.657265726572657,1
0.707968441050027,1.0813875279680312,0.47748394381891857,1.0,14.302430243024302,1
0.24645463108875193,0.11361806643738874,0.4072093477000686,1.0,13.32933293329333,1
0.2824532728706099,0.731784077174689,0.0024214867540082656,1.0,6.125612561256125,1
0.13385534190257642,0.09655220081110297,0.15285372116718193,1.0,4.935493549354935,0
0.025306246899217927,0.0738704889532309,0.163926742988616,1.0,6.3146314631463145,1
1.017839400845342,0.7378840615955254,3.1264089740442924,1.0,6.573657365736573,0
0.8474908418997188,1.1421874994661791,1.3429317659337778,1.0,8.61086108610861,1
0.9420929400295375,0.16173532898870246,1.388317598893942,1.0,9.996999699969997,1
0.3830006806472949,0.006450808756225461,0.9011137433630303,1.0,7.74977497749775,1
0.011165655393180844,0.220668957758344,0.6917905309062755,1.0,7.343734373437344,1
1.543501740482982,1.4722491560917015,0.8308174016203079,1.0,6.986698669866986,1
0.16803270916029087,3.0521630813637737,0.03508524402571821,1.0,18.13181318131813,1
2.1599455062417348,0.0016443454756151592,1.4431583362034506,1.0,4.3824382438243825,1
0.24914150677385835,0.6289922177541856,2.3185127524771802,1.0,8.743874387438744,1
0.13739886890892708,0.10774781856797644,0.35481167382045087,1.0,11.446144614461446,1
0.6373407906014511,2.847187748105618,1.4591368042456336,1.0,7.623762376237623,1
1.109732229469341,0.4055610171383578,0.01885561102032082,1.0,10.634063406340633,1
0.03186494686204995,1.7537585795238957,0.25204021431219875,1.0,8.51985198519852,1
1.6312688072399446,1.5886207744057816,3.709898616415432,1.0,4.48044804480448,1
"""), header=1)



waltons_dataset = generate_waltons_data()
regression_dataset = generate_regression_dataset()
dd_dataset = generate_dd_dataset()
lcd_dataset = generate_left_censored_data()