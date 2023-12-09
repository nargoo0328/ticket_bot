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
import datetime
 
tixcraft = True
if tixcraft:
    import torchvision.transforms as transforms
    import torch.nn.functional as F
    import torch
    from model import model, captcha

# browser_path = "C:\ProgramData\Microsoft\Windows\Start/Menu\Programs\msedgedriver.exe"
model_path = './best_model.pt'

class ticket_bot():
    def __init__(self,options,browser_type,website='tixcraft',ticket_num=1,cuda=False,login=None):
        """
            args:
                options: web browser options
                website: tixcraft or kktix
                ticket_num: Number of tickets you want to buy
                cuda: If you are using GPU to run captcha model
                login: For kktix only,
                    format: List[str{account},str{password}]
        """
        if browser_type == "Edge":
            self.browser = webdriver.Edge(options)
        elif browser_type == "Chrome":
            self.browser = webdriver.Chrome()
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

    def click_cookie(self):
        try:
            self.wait.until(EC.presence_of_element_located((By.XPATH,'/html/body/div[6]/div[2]/div/div[2]/div[1]/div/div[2]/div/div[1]/button'))).click()
            print("Clicking accept cookies...")
        except:
            pass

    def run(self,**args):
        if self.website == 'tixcraft':
            self.tixcraft(**args)
        elif self.website == 'kktix':
            self.kktix(**args)

    def login(self,token):
        if token is not None:
            self.browser.add_cookie({"name": "SID", "value": token})

    def get_average_process_time(self,times=50):
        print(f"Start refreshing pages for {times} times")
        process_time = []
        for _ in range(times):
            s = time.time()
            element = self.browser.find_element(By.XPATH,'//*[@id="tab-func"]/li[1]/a')
            element.click()
            game_list = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,'#gameList > table > tbody')))
            game_list = game_list.find_elements(By.XPATH,'tr')
            self.wait.until(EC.visibility_of(game_list[0]))
            process_time.append(time.time()-s) 
            self.get_page()
        print(f"Average process time: {np.array(process_time).mean()}")
        self.avg_time = np.array(process_time).mean()

    def check_time(self,target_time):
        date, hour, minute, second = target_time
        target_second = date * 3600 * 24 + hour*3600 + minute * 60 + second # - self.avg_time
        while True:
            current_time = datetime.datetime.now()
            tts = target_second - current_time.day *3600*24 - current_time.hour * 3600 - current_time.minute * 60 - current_time.second - current_time.microsecond/1e6 
            print(f"Current time: {current_time.hour}:{current_time.minute}:{current_time.second}.{current_time.microsecond//1000}, time to sell: {round(tts,3)} seconds",end='\r')
            if tts <0.9:
                print()
                break
            time.sleep(0.001)
        return True

    def tixcraft(self, target_time: Optional[list]=None, date: Optional[str]=None, seat_choice: Optional[list]=None, price: Optional[str]=None):

        def check_seat(text):
            for i,s in enumerate(seat_choice):
                if s in text:
                    return i
            return -1
        
        browser = self.browser
        refresh_flag = True
        wait = self.wait
        print("選場次")
        self.check_time(target_time)
        while refresh_flag:
            element = browser.find_element(By.XPATH,'//*[@id="tab-func"]/li[1]/a')
            element.click()
            # wait.until(EC.visibility_of(element)).click()

            # 選場次
            game_list = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,'#gameList > table > tbody')))
            game_list = game_list.find_elements(By.XPATH,'tr')
            wait.until(EC.visibility_of(game_list[0]))
            for game in game_list:
                game_child = game.find_elements(By.XPATH,'td')
                if date in game_child[0].text:
                    if "立即訂購" in game_child[-1].text:
                        try:
                            self.browser.execute_script("arguments[0].scrollIntoView(true);", game_child[-1])
                            time.sleep(0.2)
                            game_child[-1].click()
                        except:
                            print("PASS")
                            self.browser.execute_script("arguments[0].scrollIntoView(true);", game_child[-1])
                            time.sleep(0.5)
                            game_child[-1].click()
                        refresh_flag = False
                    else:
                        print("\tTickets are not available, refreshing...")   
                        self.get_page() 
                    break
        
        print("\n選座位")

        refresh_flag = True 
        skip_region = True if "張數" in self.browser.title else False

        while refresh_flag and not skip_region:                        
            areas_list = wait.until(EC.presence_of_element_located((By.XPATH,'/html/body/div[2]/div[1]/div[3]/div/div/div/div[2]/div[2]')))
            flag = False
            for area_list in areas_list.find_elements(By.CLASS_NAME,'area-list'):
                seats_list = area_list.find_elements(By.XPATH,'li')
                for seat in seats_list:
                    seat_index = check_seat(seat.text)
                    seat_flag = True if seat_index >=0 else False
                    if seat_flag or str(price) in seat.text:
                        print(f"\t選擇座位: {seat.text}")
                        self.actions.move_to_element(seat).perform()
                        try:
                            seat.find_element(By.XPATH,'a').click()
                            refresh_flag = False
                        except:
                            seat_choice.pop(seat_index)
                        flag = True
                        break
                if flag:
                    break
            if not flag:
                print("\t找不到目標座位，請手動點選")
                refresh_flag = False
        flag = True
        print("\n選票數")
        while flag:
            try:
                tickets_list = wait.until(EC.presence_of_element_located((By.XPATH,'/html/body/div[2]/div[1]/div[3]/div/div/div/form/div[1]/table/tbody'))).find_elements(By.CLASS_NAME,'gridc')
            except:
                print("\tWating for user...")
                continue
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
            if skip_region:
                browser.find_element(By.XPATH,'//*[@id="form-ticket-ticket"]/div[4]/button').click()
            else:
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
