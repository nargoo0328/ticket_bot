from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time
from PIL import Image
from io import BytesIO
import numpy as np
import urllib
import cv2
import math
from typing import Optional

tixcraft = False
if tixcraft:
    import torchvision.transforms as transforms
    import torch.nn.functional as F
    import torch
    from model import model, captcha

browser_path = "C:\ProgramData\Microsoft\Windows\Start/Menu\Programs\msedgedriver.exe"
model_path = 'ckpts/6_12_0_53/best_model.pt'

class ticket_bot():
    def __init__(self,options,website='tixcraft',ticket_num=1,cuda=False,login=None):
        """
            args:
                options: web browser options
                website: tixcraft or kktix
                ticket_num: Number of tickets you want to buy
                cuda: If you are using GPU to run captcha model
                login: For kktix only,
                    format: List[str{account},str{password}]
        """
        self.browser = webdriver.Edge(browser_path,options = options)
        self.wait = WebDriverWait(self.browser, 5)
        self.actions = ActionChains(self.browser)
        self.website = website
        self.ticket_num = ticket_num
        if login is not None:
            assert len(login) == 2
            if website == 'kktix':
                self.browser.get('https://kktix.com/users/sign_in?back_to=https%3A%2F%2Fkktix.com%2F')
                self.browser.find_element(By.XPATH,'/html/body/div[3]/div[2]/div/div/form/div[1]/div/input').send_keys(login[0])
                self.browser.find_element(By.XPATH,'/html/body/div[3]/div[2]/div/div/form/div[2]/div/input').send_keys(login[1])
                self.browser.find_element(By.XPATH,'/html/body/div[3]/div[2]/div/div/form/input[3]').click()


        if website == 'tixcraft':
            # captcha model
            self.cuda = cuda
            self.engine = get_engine(model_path,cuda=cuda)
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.8982),(0.1465)),
            ])
            if cuda:
                self.engine.model = self.engine.model.cuda()
            self.run_image(np.zeros((96,128)).astype(np.float32))
            

    def run_image(self,img):
        assert type(img) == np.ndarray
        img = self.transform(img)
        if self.cuda:
            img = img.cuda()
        pred = self.engine.model(img[None])
        pred = F.softmax(pred[0],-1).argmax(-1) # self.n
        output_pred = ''
        for s1 in pred:
            if s1 == 0:
                break
            output_pred += chr(96+s1)
        return output_pred
    
    def get_page(self,url=None):
        if url is not None:
            self.url = url
        else:
            url = self.url
        self.browser.get(url)

    def run(self,**args):
        if self.website == 'tixcraft':
            self.tixcraft(**args)
        elif self.website == 'kktix':
            self.kktix(**args)

    def tixcraft(self, date: Optional[str]=None, seat_choice: Optional[str]=None, price: Optional[str]=None):
        assert date is not None and (price is not None or seat_choice is not None)
        browser = self.browser
        refresh_flag = True
        wait = self.wait
        print("選場次")
        while refresh_flag:
            browser.find_element(By.XPATH,'//*[@id="tab-func"]/li[1]/a').click()

            # 選場次
            game_list = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,'#gameList > table > tbody')))
            game_list = game_list.find_elements(By.XPATH,'tr')
            for game in game_list:
                wait.until(EC.visibility_of(game))
                game_child = game.find_elements(By.XPATH,'td')
                if date in game_child[0].text:
                    try:
                        game_child[-1].find_element(By.XPATH,'button')
                        game_child[-1].click()
                        refresh_flag = False
                    except:
                        print("Tickets are not available, refreshing...",end='\r')   
                        self.get_page() 
                    break
        
        print("\n選座位")                                   
        areas_list = wait.until(EC.presence_of_element_located((By.XPATH,'/html/body/div[2]/div[1]/div[3]/div/div/div/div[2]/div[2]')))
        flag = False
        for area_list in areas_list.find_elements(By.CLASS_NAME,'area-list'):
            seats_list = area_list.find_elements(By.XPATH,'li')
            for seat in seats_list:
                if str(seat_choice) in seat.text or str(price) in seat.text:
                    seat.find_element(By.XPATH,'a').click()
                    flag = True
                    break
            if flag:
                break
        assert flag == True
        while flag:
            print("選票數")
            tickets_list = wait.until(EC.presence_of_element_located((By.XPATH,'/html/body/div[2]/div[1]/div[3]/div/div/div/form/div[1]/table/tbody'))).find_elements(By.CLASS_NAME,'gridc')

            # default index 0 (standard ticket)
            ticket = tickets_list[0]
            ticket_option = Select(ticket.find_element(By.XPATH,'.//*/select'))
            ticket_option.select_by_value(str(self.ticket_num))
            
            img = browser.find_element(By.XPATH,'//*[@id="TicketForm_verifyCode-image"]').screenshot_as_png
            img = url_to_image(img,method=2)
            pred = self.run_image(img)

            # submit captcha prediction
            browser.find_element(By.XPATH,'//*[@id="TicketForm_verifyCode"]').send_keys(pred)

            # click agreement
            browser.find_element(By.XPATH,'//*[@id="TicketForm_agree"]').click()

            # submit
            browser.find_element(By.XPATH,'//*[@id="form-ticket-ticket"]/div[4]/button[2]').click()
            try:
                alert = browser.switch_to.alert
                alert.accept()
                print("Captcha wrong. Try again...")
            except:
                flag = False
        return    

    def kktix(self,seat_choice: Optional[str]=None,price: Optional[int]=None,**kwargs):
        assert price is not None or seat_choice is not None
        if price is not None :
            if math.log(price,10) >= 3:
                price = str(price)
                price = price[:-3] + ',' +price[-3:]
        wait = self.wait
        browser = self.browser                   
        refresh_flag = None
        ticket_flag = False
        while True:
            wait.until(EC.element_to_be_clickable((By.XPATH,'/html/body/div[3]/div[4]/div/div/div[1]/div/div[2]/div/div[3]/button'))).click()
            try:    
                ticket_root = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,'#registrationsNewApp > div > div:nth-child(5) > div.ticket-list-wrapper.ng-scope > div.ticket-list.ng-scope')))#browser.find_element(By.CSS_SELECTOR,'#registrationsNewApp > div > div:nth-child(5) > div.ticket-list-wrapper.ng-scope > div.ticket-list.ng-scope') 
            except:
                print("Try with 'seat'")
                ticket_root = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,'#registrationsNewApp > div > div:nth-child(5) > div.ticket-list-wrapper.ng-scope.with-seat > div.ticket-list.ng-scope')))#browser.find_element(By.CSS_SELECTOR,'#registrationsNewApp > div > div:nth-child(5) > div.ticket-list-wrapper.ng-scope.with-seat > div.ticket-list.ng-scope')
            tickets = ticket_root.find_elements(By.XPATH,'.//*')                     
            for ticket in tickets:
                if ticket.get_attribute('class') == 'ticket-unit ng-scope':
                    self.actions.move_to_element(ticket).perform()
                    ticket_info = ticket.find_element(By.CLASS_NAME,'display-table-row')
                    # ticket_type = ticket.find_element(By.CLASS_NAME,'ticket-name ng-binding')
                    # current_price = ticket.find_element(By.CLASS_NAME,'ticket-price')
                    if str(price) in ticket_info.text or str(seat_choice) in ticket_info.text:
                        print(f"Found ticket: {price}",end='\r')
                        ticket_flag = True
                        break
            if not ticket_flag:
                print("Can not find target price. Choosing top price")
                ticket = tickets[0]
            for tmp in ticket.find_elements(By.XPATH,'.//*'):
                if tmp.get_attribute('class') == 'ticket-quantity ng-scope':
                    refresh_flag = True
                    break
            if refresh_flag:
                break
            else:
                print("Ticket is not ready. Refreshing",end='\r')
                self.get_page()
        tmp.find_element(By.CSS_SELECTOR,'input').send_keys([str(self.ticket_num)])
                                      #
        browser.find_element(By.CSS_SELECTOR,'#person_agree_terms').click() 
        browser.find_element(By.CSS_SELECTOR,'#registrationsNewApp > div > div:nth-child(5) > div.form-actions.plain.align-center.register-new-next-button-area > button').click() 
                                       
            
def get_engine(model_path,cuda=False):
    net = model.captcha_model(64,cuda=cuda)
    net = net.eval()
    engine = captcha.Captcha(6,net)
    device = torch.device('cuda') if cuda else torch.device('cpu')
    engine.model.load_state_dict(torch.load(model_path,map_location=device))
    return engine

def url_to_image(url,method=1):
    if method == 1:
        with urllib.request.urlopen(url) as resp:
            s = resp.read()
        image = np.asarray(bytearray(s), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image
    else:
        # img = url.replace('data:image/png;base64,', '')
        # img = base64.b64decode(img)
        img = Image.open(BytesIO(url))
        img = img.convert('L')
        img = np.asarray(img, dtype="uint8")
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(128,96),interpolation=cv2.INTER_NEAREST)
        return img
