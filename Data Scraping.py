import time
from selenium import webdriver
from bs4 import BeautifulSoup as bs

def getVarietalURl():
    browser = webdriver.Chrome()
    url = 'http://www.klwines.com/Wines'
    browser.get(url)
    browser.execute_script("loadEndecaRefinements('System.Web.HttpCookie', '4', '2', '', '8', 'False', '', '57', 'True');")
    time.sleep(2)
    innerHTML = browser.execute_script("return document.body.innerHTML")
    soup = bs(innerHTML,'html.parser')
    div = soup.find('div',attrs={'id':'endecaRefinements'})
    list_of_varietals = div.ul.find_all('a')
    list_of_varietals = list_of_varietals[1:]
    varietals_url = {}
    for i in list_of_varietals:
        varietals_url[i.text] = 'http://www.klwines.com' + i['href']
    return (varietals_url,browser)

def getAllBrands(varietalUrl,browser):
    browser.get(varietalUrl)
    innerHTML = browser.execute_script("return document.body.innerHTML")
    soup = bs(innerHTML,'html.parser')
    div = soup.find_all('div',attrs={'class':'result clearfix'})
    thisVarietalBrands = []
    for i in div:
        thisVarietalBrands.append('http://www.klwines.com'+i.find('a')['href'])
    pages = soup.find('div',attrs={'class':'floatLeft'}).find_all('a')
    if(pages[-1].text=='next' or pages[-2].text=='next'):
        nextPage = 'http://www.klwines.com'
        if(pages[-1].text=='next'):nextPage+=pages[-1]['href']
        else:nextPage+=pages[-2]['href']
        thisVarietalBrands += getAllBrands(nextPage,browser)
    return thisVarietalBrands

def getReviews(wineUrl,browser):
    browser.get(wineUrl)
    innerHTML = browser.execute_script("return document.body.innerHTML")
    soup = bs(innerHTML,'html.parser')
    reviews_div = soup.find_all('div',attrs={'class':'ReviewTextDetail'})
    reviews = [(i.text).strip() for i in reviews_div]
    attr = [(i.text).strip() for i in soup.find_all(attrs={'class':'detail_td1'})]
    values = [(i.text).strip() for i in soup.find_all(attrs={'class':'H3ProductDetail_detail'})]
    if(attr[-1]=='Alcohol Content (%):'):
        values += (soup.find(attrs={'class':'addtl-info-block'}).table.tbody.find_all('td',attrs={'class':'detail_td'})[-1].text).strip()
    wineInfo = dict(zip(attr,values))
    wineInfo['Reviews'] = reviews
    wineInfo['name']=(soup.find('h1').text).strip()
    return wineInfo

