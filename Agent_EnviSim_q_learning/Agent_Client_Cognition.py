# Neste arquivo é onde você cria uma 'mente' para seu agente.

import random
import json
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from datetime import datetime

from Agent_Client_Setup import Stt, InpSensors, OutNeurons

from Agent_Client_Setup import keyMagACT, keyMagMOV, keyMagREQ, keyMagROT, ACTgrb, ACTlev, ACTnil, \
    MOVfor, REQfwd, REQlft, REQl45, REQori, REQrst, REQrgt, REQr45, ROTlft, ROTrgt, ROTbck, \
    keyMwpSNS, keyMwpCOL, keyMwpOUT, keyMwpSRV, keyMwpPHR, keyMwpPOS, keyMwpDIR, keyMwpDVA, \
    SNSbrz, SNSdng, SNSfsh, SNSgol, SNSini, SNSobs, SNStch, SNSnth, CLDbnd, CLDobs, CLDwll, \
    OUTcnt, OUTdie, OUTgrb, OUTnon, OUTrst, OUTsuc, SRVcnn, SRVinv, SRVnor, SRVpsd, \
    DIRn, DIRne, DIRe, DIRse, DIRs, DIRsw, DIRw, DIRnw


# este método é usado para 'analisar a resposta/feedback' recebido do EnviSim
def feedback_analysis(vecInpSens: np.int32, carryRWD: int) -> int:
    outy = -1  # por default, o índice de saída é um índice de erro
    if np.sum(vecInpSens) != len(vecInpSens):  # se o número de bits for '!= 1, 'inferir' retornará um erro (-1)
        return outy
    else:
        inx = np.argmax(vecInpSens)  # isso obtém o índice do bit ativo dentro do vetor de feedback
        tmpStr: str = InpSensors[inx]
        if tmpStr == 'inp_' + SNSgol and carryRWD == 0:
            outy = OutNeurons.index("out_act_grab")
        elif tmpStr == 'inp_' + SNSini and carryRWD == 1:
            outy = OutNeurons.index("out_act_leave")
        elif tmpStr == 'inp_' + OUTgrb:
            outy = 50
        elif tmpStr == 'inp_' + OUTsuc and carryRWD == 1:
            outy = 100
        elif tmpStr == 'inp_' + OUTdie:
            outy = -100
        else:
            outy = OutNeurons.index("out_act_nill")
    return outy


# MÉTODO NO QUAL VOCÊ VAI INSERIR INTELIGÊNCIA NO AGENTE !!!
# este método é usado para 'inferência', ou seja, para tomar decisões

## Global
qtable = np.zeros((125, 4))
#qtable_treinada = np.load('nome_do_arquivo.npy')
# Criando um dicionário para armazenar as combinações e seus índices
comb_dict = {}
indice = 0
# Loop para variar a primeira variável
for var1 in range(1, 6):
    # Loop para variar a segunda variável
    for var2 in range(1, 6):
        # Loop para variar a terceira variável
        for var3 in range(1, 6):
            # Mapeando a combinação para o índice no dicionário
            comb_dict[(var1, var2, var3)] = indice
            indice += 1

state = 0
done = False
cont = 0
episodes = 1
num_iter = 505
energia = 500
intervalo_display = 500
# Hyperparameters
alpha = 0.5  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Amount of randomness in the action selection
epsilon_decay = 0.001  # Fixed amount to decrease
# List of outcomes to plot
outcomes = np.zeros(num_iter)
numero_passos = np.zeros(num_iter)

def resume_entrada(leitura):
    resum_entry = 1
    if leitura in [0, 5]:  # nothing, none
        resum_entry = 1
    elif leitura in [3, 8, 9, 11]:  # qq um que tenha flash
        resum_entry = 2
    elif leitura in [1, 7, 10]:  # qq um que tenha breeze ou stench
        resum_entry = 3
    elif leitura in [2, 6, 12, 15]:  # qq um que tenha obstáculo
        resum_entry = 4
    elif leitura in [4]:  # encontrou goal
        resum_entry = 5
    return resum_entry

###################### Q Learning ###########################

def infer(vecInpSens: np.int32) -> int:
    global qtable, intervalo_display, state, done, alpha, gamma, epsilon, epsilon_decay, cont, episodes, num_iter, energia, comb_dict, outcomes
    decisao = [3, 11, 12, 13]    # Frente / Esquerda / Direita/ Tras
    randon_decisao = [0, 0, 0, 1, 1, 2, 2, 3]
    leitura_frente = np.nonzero(vecInpSens[0])[0]
    #print('leitura_frente', leitura_frente)
    leitura_direita = np.nonzero(vecInpSens[1])[0]
    #print('leitura_direita', leitura_direita)
    leitura_esquerda = np.nonzero(vecInpSens[2])[0]
    #print('leitura_esquerda', leitura_esquerda)
    resum_entry_f = resume_entrada(leitura_frente)
    #print('resumo_frente', resum_entry_f)
    resum_entry_d = resume_entrada(leitura_direita)
    #print('resumo_direita', resum_entry_d)
    resum_entry_e = resume_entrada(leitura_esquerda)
    #print('resumo_esquerda', resum_entry_e)
    # Training
    # Generate a random number between 0 and 1
    rnd = np.random.random()
    if not done and cont < energia and episodes < num_iter:
        cont = cont + 1
        numero_passos[episodes] = cont
        # If random number < epsilon, take a random action
        if rnd < epsilon:
            action = randon_decisao[random.randint(0, 7)]
        # Else, take the action with the highest value in the current state
        else:
            action = np.argmax(qtable[state])
        # Implement this action and move the agent in the desired direction
        new_state = comb_dict[(resum_entry_f, resum_entry_d, resum_entry_e)]
        #print('new_satate', new_state)
        # Reward function
        if leitura_frente in [0, 5, 18]:  # nothing, none
            reward = 0
        elif leitura_frente in [3, 8, 9, 11]:  # qq um que tenha flash a frente
            reward = 0.1
        elif leitura_frente in [1, 7, 10]:  # qq um que tenha breeze ou stench
            reward = -0.5
        elif leitura_frente in [2, 6, 12, 13, 14, 16]:  # qq um que tenha obstáculo
            reward = -2
        elif leitura_frente in [4]:  # encontrou goal
            reward = 50
            done = True
            print('Encontrou o ouro')
            outcomes[episodes] = 1
        else:
            reward = 0
        #print('reward ', reward)
        # Update Q(s,a)
        qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])
        # Update our current state
        state = new_state
    elif episodes < num_iter:
        epsilon = max(epsilon - epsilon_decay, 0)
        done = False
        cont = 0
        episodes = episodes + 1
        print('Episodios: ', episodes)
        # Update epsilon
        epsilon = max(epsilon - epsilon_decay, 0)
        return 8
    if episodes % intervalo_display == 0 and cont == 1:
        plt.figure(figsize=(12, 5))
        plt.xlabel("Run number")
        plt.ylabel("Outcome")
        ax = plt.gca()
        ax.set_facecolor('#efeeea')
        plt.bar(range(len(outcomes)), outcomes, color="#0A047A", width=1.0)
        plt.show()
        plt.figure(figsize=(12, 5))
        plt.xlabel("Run number")
        plt.ylabel("Numero de passos")
        ax = plt.gca()
        ax.set_facecolor('#efeeea')
        plt.bar(range(len(numero_passos)), numero_passos, color="#0A047A", width=1.0)
        plt.show()
        print(qtable)
        # Obter a data e hora atual
        data_hora_atual = datetime.now()
        formato_data_hora = data_hora_atual.strftime('%Y-%m-%d_%H-%M-%S')  # Formato: AAAA-MM-DD_HH-MM-SS
        # Nome do arquivo com a data e hora atual
        nome_arquivo1 = f'q_table_energia_{energia}_iteracao_{episodes}_hora_{formato_data_hora}.npy'
        # Salvar o numpy array no arquivo com o nome incluindo a data e hora
        np.save(nome_arquivo1, qtable)
        # Nome do arquivo com a data e hora atual
        nome_arquivo2 = f'outcome_{formato_data_hora}.npy'
        # Salvar o numpy array no arquivo com o nome incluindo a data e hora
        np.save(nome_arquivo2, outcomes)
        # Nome do arquivo com a data e hora atual
        nome_arquivo3 = f'numero_de_passos_{formato_data_hora}.npy'
        # Salvar o numpy array no arquivo com o nome incluindo a data e hora
        np.save(nome_arquivo3, numero_passos)
        '''energia = energia + 5
        print('Energia', energia)'''

    return decisao[action]
# este método cria uma msg para o EnviSim solicitando informações do Wumpus World
# input: indx de uma msg a ser enviada, e a distância da posição atual na grade
def create_msg(indx_out: int, dist: int) -> str:
    rasc: str = OutNeurons[indx_out]
    msg = ''  # local var msg starts empty
    if rasc == 'out_req_' + REQfwd:
        msg = '{\"' + keyMagREQ + '\":[\"' + REQfwd + '\",' + str(dist) + ']}'  # solicitar info 'dist' posição(ões) adiante
    elif rasc == 'out_req_' + REQlft:
        msg = '{\"' + keyMagREQ + '\":[\"' + REQlft + '\",' + str(dist) + ']}'  # request info 90 deg left, 'dist' position(s)
    elif rasc == 'out_req_' + REQl45:
        msg = '{\"' + keyMagREQ + '\":[\"' + REQl45 + '\",' + str(dist) + ']}'  # request info 45 deg left, 'dist'  position(s)
    elif rasc == 'out_req_' + REQori:
        msg = '{\"' + keyMagREQ + '\":[\"' + REQori + '\",' + str(dist) + ']}'  # request orientation (angle from guidingStar)
    elif rasc == 'out_req_' + REQrst:
        msg = '{\"' + keyMagREQ + '\":[\"' + REQrst + '\",' + str(0) + ']}'  # solicite ao EnviSim reiniciar a missão
    elif rasc == 'out_req_' + REQrgt:
        msg = '{\"' + keyMagREQ + '\":[\"' + REQrgt + '\",' + str(dist) + ']}'  # request info 90 deg right, 'dist' position(s)
    elif rasc == 'out_req_' + REQr45:
        msg = '{\"' + keyMagREQ + '\":[\"' + REQr45 + '\",' + str(dist) + ']}'  # request info 45 deg right, 'dist' position(s)
    elif rasc == 'out_act_' + ACTgrb:
        msg = '{\"' + keyMagACT + '\":[\"' + ACTgrb + '\",' + str(dist) + ']}'  # ação: pegue o ouro
    elif rasc == 'out_act_' + ACTlev:
        msg = '{\"' + keyMagACT + '\":[\"' + ACTlev + '\",' + str(dist) + ']}'  # ação: sair da caverna
    elif rasc == 'out_act_' + ACTnil:
        msg = '{\"' + keyMagACT + '\":[\"' + ACTnil + '\",' + str(dist) + ']}'  # action: nill, do nothing for while
    elif rasc == 'out_mov_' + MOVfor:
        msg = '{\"' + keyMagMOV + '\":[\"' + MOVfor + '\",' + str(dist) + ']}'  # avançar 'dist' posição(ões)
    elif rasc == 'out_rot_' + ROTlft:
        msg = '{\"' + keyMagROT + '\":[\"' + ROTlft + '\",' + str(2) + ']}'  # girar para a esquerda 2 x 45 graus
    elif rasc == 'out_rot_' + ROTrgt:
        msg = '{\"' + keyMagROT + '\":[\"' + ROTrgt + '\",' + str(2) + ']}'  # girar para a direita 2 x 45 graus
    elif rasc == 'out_rot_' + ROTbck:
        msg = '{\"' + keyMagROT + '\":[\"' + ROTbck + '\",' + str(4) + ']}'  # girar para a trás = 4 x 45 graus

    return msg


# este método interpreta a mensagem do EnviSim
# o fn retorna um novo estado para o FSM principal, um código de string (ou '') e o índice do sensor de entrada detectado
def interpreting(envisim_answ: str) -> tuple[Stt, str, int, np.int32]:
    jobj = json.loads(envisim_answ)  # 1o. torne a string recebida um objeto Json
    str_code = ''  # inicia o strCode vazio
    stt_mm = Stt.DECIDING  # por default, o próximo estado da main-FSM é DECIDING
    idx_inp_sns: int = 0  # default para o índice da entrada é zero
    CurrSensBits = np.zeros(32, dtype=np.int32)  # todos os sensores com flag = 0

    # 1). teste se a msg tem a chave 'servidor' - são mensagens prioritárias
    if keyMwpSRV in jobj:  # se o EnviSim enviou a chave keyMwpSRV ('server')
        jrasc = jobj[keyMwpSRV]  # jrasc contém o payload da mensagem (sem a chave)
        if SRVcnn in jrasc:  # após receber uma msg 'connected', o próximo passo é:
            # print('after connecting, ask EnviSim to restart...')
            str_code = SRVcnn  # string com a msg SRVcnn
            stt_mm = Stt.RESTARTING  # para REINICIAR o EnviSim (novo estado = RESTARTING)
        elif SRVinv in jrasc:  # O EnviSim disse que a última msg era inválida
            #  print('EnviSim send invalid msg...')
            str_code = 'msg_invalid'  # o código de erro para msg 'invalid'
            stt_mm = Stt.ERRORS  # mudança para estado que lida com ERROS
        elif SRVpsd in jrasc:  # EnviSim disse que o servidor está pausado
            #  print('EnviSim is paused...')
            str_code = 'server_paused'  # o código de erro para a mensagem 'paused'
            stt_mm = Stt.ERRORS  # mudança para estado que lida com ERROS
        elif SRVnor in jrasc:  # EnviSim disse que o servidor está em operação normal
            #  print('EnviSim in normal operation...')
            str_code = 'server_normal'  # código para a msg 'normal'
            stt_mm = Stt.ERRORS  # mudança para estado que lida com ERROS

    # 2). se a msg tiver a chave 'outcome', o agente: 
    # não pode, morreu, agarrou, reiniciou, sucesso (cannot, died, grabbed, restarted, success)
    # algumas mgs colocam a main-FSM no estado EXCEPTIONS outros colocam a máquina em operação normal
    elif keyMwpOUT in jobj:  # se o EnviSim enviou a chave keyMwpOUT ('outcome')
        jrasc = jobj[keyMwpOUT]  # jrasc contém o payload da mensagem (sem a chave)
        if OUTrst in jrasc:  # outcome = 'restarted' o EnviSim reiniciou uma missão
            # print('--> EnviSim has restarted a mission')
            idx_inp_sns = InpSensors.index('inp_' + OUTrst)  # index para o 'restarted input sensor'
            CurrSensBits[idx_inp_sns] |= 0b1  # ajusta o flag-bit p/ 1
            str_code = 'inp_' + OUTrst  # retorne também a string 'inp_restarted'
        elif OUTgrb in jrasc:  # se jrasc='grabbed', o agente segura a recompensa (ouro)
            # print('Good job: the agent grabbed the REWARD...')
            idx_inp_sns = InpSensors.index('inp_' + OUTgrb)  # o código interno p/ GRABBED
            CurrSensBits[idx_inp_sns] |= 0b1  # ajusta o bit para 1
        elif OUTdie in jrasc:  # se jrasc='died', to agente morreu (missão terminada)
            # print('Bad news: the agent DIED...')
            idx_inp_sns = InpSensors.index('inp_' + OUTdie)  # código interno p/ DIED
            CurrSensBits[idx_inp_sns] |= 0b1  # ajusta o flag-bit para 1
            str_code = OUTdie  # DIED (MORREU) 	é uma exceção - fim da missão
            stt_mm = Stt.EXCEPTIONS  # muda o estado p/ EXCEPTIONS p/ esse 'outcome'
        elif OUTsuc in jrasc:  # se jrasc='success', o agente venceu (completou a missão)
            # print('SUCCESS: the agent completed the mission...')
            idx_inp_sns = InpSensors.index('inp_' + OUTsuc)  # código interno p/ SUCCESS
            CurrSensBits[idx_inp_sns] |= 0b1  # ajusta o flag-bit para 1
            str_code = OUTsuc  # sucesso é uma exceção - fim da missão
            stt_mm = Stt.EXCEPTIONS  # muda o estado p/ EXCEPTIONS para esse 'outcome'
        elif OUTcnt in jrasc:  # if vem a msg 'cannot' o último comando NÃO pode ser executado
            # print('Oops: the agent CANNOT do the last action...')
            idx_inp_sns = InpSensors.index('inp_' + OUTcnt)  # código p/ CANNOT
            CurrSensBits[idx_inp_sns] |= 0b1  # ajusta o flag-bit para 1
        elif OUTnon in jrasc:  # se veio 'none', o último comando resultou em ação nenhuma
            # print('Ooops: the last command caused no action...')
            idx_inp_sns = InpSensors.index('inp_' + OUTnon)  # codigo para NO ACTIONS
            CurrSensBits[idx_inp_sns] |= 0b1  # ajusta o flag-bit para 1
        else:  # 'undefined', EnviSim está dizendo que não identificou sua msg
            #  print('Attention: outcome came - UNDEFINED - ?!')
            str_code = 'undefined_outcome'  # erro: msg não identificada com key 'outcome'
            stt_mm = Stt.ERRORS  # muda o estado da FSM para lidar com ERROS

    # 3). se a msg tem a key 'collision', o agente colidiu ou vai colidir com algo
    # COLLISION significa que o último comando NÃO foi executado pelo EnviSim
    elif keyMwpCOL in jobj:
        jrasc = jobj[keyMwpCOL]  # jrasc = payload da msg (sem a key)
        if CLDbnd in jrasc:  # indica uma colisão com as bordas do Wumpus
            idx_inp_sns = InpSensors.index('inp_' + CLDbnd)  # código p/ collision com as bordas
            CurrSensBits[idx_inp_sns] |= 0b1  # ajusta o flag-bit para 1
        elif CLDobs in jrasc:
            idx_inp_sns = InpSensors.index('inp_' + CLDobs)  # código p/ collision com obstáculos dentro do mundo
            CurrSensBits[idx_inp_sns] |= 0b1  # ajusta o flag-bit para 1
        elif CLDwll in jrasc:
            idx_inp_sns = InpSensors.index('inp_' + CLDwll)  # código p/ collision com as paredes dentro da cena
            CurrSensBits[idx_inp_sns] |= 0b1  # ajusta o flag-bit para 1
        else:  # 'undefined', EnviSim responde algo indefinido
            print('Attention: collision came - undefined - ?!')
            str_code = 'undefined_collision'  # error: msg inesperada com key 'collision'
            stt_mm = Stt.ERRORS  # muda o estado da FSM para lidar com ERROS

    # 4). se a mensagem tiver a chave 'sense', o EnviSim informa o que o agente experimentou na posição requerida
    # o agente pode 'sentir': brisa, perigo, flash, objetivo, inicial, obstrução, fedor
    # duas ou mais 'sensações' podem vir como carga útil em uma mensagem - ou pode vir vazio, [], sem sentido!
    elif keyMwpSNS in jobj:
        jrasc = jobj[keyMwpSNS]  # jrasc contém a carga útil da mensagem sense (nenhuma, 1 ou mais sensações)
        if len(jrasc) == 3:
            if (SNSbrz in jrasc) and (SNSfsh in jrasc) and (SNStch in jrasc):
                idx_inp_sns = InpSensors.index('inp_bfs')  # sensores Breeze, Flash e Stench estão ativos
                CurrSensBits[idx_inp_sns] |= 0b1  # define esse bit de sinalização como um
        elif len(jrasc) == 2:
            if (SNSbrz in jrasc) and (SNSfsh in jrasc):
                idx_inp_sns = InpSensors.index('inp_bf')  # sensores Breeze e Flash estão ativos
                CurrSensBits[idx_inp_sns] |= 0b1  # define esse bit de sinalização como um
            elif (SNSbrz in jrasc) and (SNStch in jrasc):
                idx_inp_sns = InpSensors.index('inp_bs')  # sensores Breeze e Stench estão ativos
                CurrSensBits[idx_inp_sns] |= 0b1  # define esse bit de sinalização como um
            elif (SNSfsh in jrasc) and (SNStch in jrasc):
                idx_inp_sns = InpSensors.index('inp_fs')  # sensores Flash e Stench estão ativos
                CurrSensBits[idx_inp_sns] |= 0b1  # define esse bit de sinalização como um
        elif len(jrasc) > 0:  # verifica se há sensores ativados
            for item in jrasc:  # para cada sensor ativo
                if SNSfsh in item:  # se o sensor Flash estiver ativo
                    idx_inp_sns = InpSensors.index('inp_' + SNSfsh)  # o índice do sensor Flash é identificado
                    CurrSensBits[idx_inp_sns] |= 0b1  # este bit de flag é definido como 1
                elif SNSdng in item:  # se o sensor Danger estiver ativo
                    idx_inp_sns = InpSensors.index('inp_' + SNSdng)  # o índice do sensor Danger é identificado
                    CurrSensBits[idx_inp_sns] |= 0b1  # este bit de flag é definido como 1
                elif SNSobs in item:  # se o sensor Obstruction estiver ativo
                    idx_inp_sns = InpSensors.index('inp_' + SNSobs)  # o índice do sensor Obstruction é identificado
                    CurrSensBits[idx_inp_sns] |= 0b1  # este bit de flag é definido como 1
                elif SNSgol in item:  # se o sensor Goal estiver ativo
                    idx_inp_sns = InpSensors.index('inp_' + SNSgol)  # o índice do sensor Goal é identificado
                    CurrSensBits[idx_inp_sns] |= 0b1  # este bit de flag é definido como 1
                elif SNSini in item:  # se o sensor Initial estiver ativo
                    idx_inp_sns = InpSensors.index('inp_' + SNSini)  # o índice do sensor Initial é identificado
                    CurrSensBits[idx_inp_sns] |= 0b1  # este bit de flag é definido como 1
                elif SNSbrz in item:  # se o sensor Breeze estiver ativo
                    idx_inp_sns = InpSensors.index('inp_' + SNSbrz)  # o índice do sensor Breeze é identificado
                    CurrSensBits[idx_inp_sns] |= 0b1  # este bit de flag é definido como 1
                elif SNStch in item:  # se o sensor Stench estiver ativo
                    idx_inp_sns = InpSensors.index('inp_' + SNStch)  # o índice do sensor Stench é identificado
                    CurrSensBits[idx_inp_sns] |= 0b1  # este bit de flag é definido como 1
        else:  # a mensagem chegou vazia: o agente não sente nada, a posição na grade está vazia
            idx_inp_sns = InpSensors.index('inp_' + SNSnth)  # o sensor Nothing é identificado
            CurrSensBits[idx_inp_sns] |= 0b1  # este bit de flag é definido como 1

        # print('idx_inp_sns: ', idx_inp_sns)  # apenas imprime o sensor de entrada identificado
        
    # 5). a mensagem pode ter a chave 'direction', isso é opcional (definido no EnviSim)
    if keyMwpDIR in jobj:
        jrasc = jobj[keyMwpDIR]  # jrasc contém a carga útil da mensagem
        if DIRn in jrasc:  # dir = 'norte', o agente está voltado para o topo da tela
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRn)  # neurônio disparado (inp_dir_n)
            CurrSensBits[idx_inp_sns] |= 0b1  # define esse bit de sinalização como um
        elif DIRne in jrasc:  # se dir = 'nordeste'
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRne)  # neurônio disparado (inp_dir_ne)
            CurrSensBits[idx_inp_sns] |= 0b1  # define esse bit de sinalização como um
        elif DIRe in jrasc:  # dir = 'leste', o agente está voltado para o lado direito da tela
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRe)  # neurônio disparado (inp_dir_e)
            CurrSensBits[idx_inp_sns] |= 0b1  # define esse bit de sinalização como um
        elif DIRse in jrasc:  # se dir = 'sudeste'
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRse)  # neurônio disparado (inp_dir_se)
            CurrSensBits[idx_inp_sns] |= 0b1  # define esse bit de sinalização como um
        elif DIRs in jrasc:  # dir = 'sul', o agente está voltado para baixo na tela
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRs)  # neurônio disparado (inp_dir_s)
            CurrSensBits[idx_inp_sns] |= 0b1  # define esse bit de sinalização como um
        elif DIRsw in jrasc:  # se dir = 'southwest'
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRsw)  # neurônio ativo (inp_dir_sw)
            CurrSensBits[idx_inp_sns] |= 0b1  # set this flag bit to one
        elif DIRw in jrasc:  # dir = 'oeste', o agente está voltado para a esquerda da tela
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRw)  # neurônio ativo (inp_dir_w)
            CurrSensBits[idx_inp_sns] |= 0b1  # set this flag bit to one
        elif DIRnw in jrasc:  # se dir = 'noroeste'
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRnw)  # neurônio ativo (inp_dir_nw)
            CurrSensBits[idx_inp_sns] |= 0b1  # set this flag bit to one
        else:  # ocorreu um erro - EnviSim enviou uma mensagem inválida
            print('Atenção: DIRECTION veio - indefinido - ?!')
            str_code = 'direcao_indefinida'  # código de erro para uma mensagem de direção incorreta
            stt_mm = Stt.ERRORS  # alterar para o estado que lida com erros

    # 6). a mensagem pode ter a chave 'pheromone', mas é opcional (definido em EnviSim)
    # a carga útil (payload) dessa mensagem tem o valor da feromônio na posição atual do agente
    if keyMwpPHR in jobj:
        jrasc = jobj[keyMwpPHR]  # jrasc tem a carga útil da mensagem
        if len(jrasc) != 1:
            print('Atenção: PHEROMONE chegou - indefinido - ?!')
            str_code = 'feromônio_indefinido'  # erro: mensagem de feromônio incorreta
            stt_mm = Stt.ERRORS  # muda para o estado que lidará com ERROS
        else:
            pherom = jrasc[0]  # obtém o valor da feromônio na posição da grade
            idx_inp_sns = InpSensors.index('inp_' + keyMwpPHR)  # um código que indica que uma posição foi capturada
            CurrSensBits[idx_inp_sns] |= 0b1  # define esse bit de sinalização como um (1)

    # 7). a mensagem pode ter a chave 'deviation', isso é opcional (configurado em EnviSim)
    # a carga da mensagem tem o ângulo entre a direção do agente e a Estrela guia do EnviSim
    if keyMwpDVA in jobj:
        jrasc = jobj[keyMwpDVA]  # jrasc tem a carga da mensagem
        if len(jrasc) != 1:
            print('Atenção: ângulo DEVIATION veio - indefinido - ?!')
            str_code = 'desvio_indefinido'  # erro: mensagem de desvio errada
            stt_mm = Stt.ERRORS  # mudança para o estado que lida com ERROS
        else:
            devAngle = jrasc[0]  # o valor atual de devAngle
            idx_inp_sns = InpSensors.index('inp_' + keyMwpDVA)  # o código após a captura de uma mensagem 'ângulo de desvio'
            CurrSensBits[idx_inp_sns] |= 0b1  # define este bit de sinalizador como um

    # 8). a mensagem pode ter a chave 'position', isso é opcional (definido em EnviSim)
    # obter a posição (x,y) do agente no mapa - se quisermos rastrear seu caminho
    if keyMwpPOS in jobj:
        stt_mm = Stt.EXCEPTIONS  # estado padrão = EXCEPTIONS para esse tipo de mensagem
        jrasc = jobj[keyMwpPOS]  # jrasc contém a carga útil da mensagem
        if len(jrasc) != 2:
            print('Atenção: POSIÇÃO recebida - indefinida - ?!')
            str_code = 'posição_indefinida'  # o código de erro para uma mensagem de posição incorreta
            stt_mm = Stt.ERRORS  # altera para o estado que lida com ERROS
        else:
            posX = jrasc[0]  # o valor atual de x
            posY = jrasc[1]  # o valor atual de y
            str_code = keyMwpPOS  # um código que indica que uma posição foi capturada
            CurrSensBits[idx_inp_sns] |= 0b1  # definir esse bit de sinalizador como um
    # print('curr out: ', CurrSensBits)
    return stt_mm, str_code, idx_inp_sns, CurrSensBits
