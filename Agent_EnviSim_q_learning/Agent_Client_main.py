#!/usr/bin/env python
# coding: utf-8

import socket
import time
import csv
import keyboard as kb

# ------------------------ setup -----------------------------
from Agent_Client_Cognition import *  # importa todos os métodos/funções de Cognition

# Nota: todas as definições de vars e inicializações estão no arquivo 'Agent_Client_Setup.py'
from Agent_Client_Setup import Stt, SubStt, InfoReqSeq, sttMM, sttSUBfsm, msg, answES, \
    energy, carryRWD, iterNum, strCode, InpSensors, idxInpSensor, nofInfoRequest, cntNofReqs,\
    delaySec, keyMagREQ, REQfwd, REQrst, keyMwpPOS, OUTdie, OUTrst, OUTsuc, posX, posY, \
    modoDados, nofWandSteps, subSttWand, seqWand, arqv_csv
# ---(end)---------------- setup -----------------------------

# ------ MAIN = here starts the main program  ------

# 1. estabelecendo a conexão de IPC 'Comunicação entre Processos' com o processo EnviSim
host_name = socket.gethostname()  # obter o nome deste computador
host_IP = socket.gethostbyname(host_name)  # obter o endereço IP deste computador (intranet)
IPC_port = 15051  # número do PORT (use o mesmo número de PORTA no programa EnviSim)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # criar um socket = sock
server_address = (host_IP, IPC_port)  # a variável endereço_servidor contém: o IP + porta_IPC
# apenas para teste: imprimir a variável endereço_servidor
# print('server: {} - port: {}'.format(host_IP,IPC_port))
# print('server: {}'.format(server_address))

# 2. loop até que a tecla 'esc' seja pressionada (ESC termina este processo)!

while msg != 'esc':

    if kb.is_pressed('esc'):  # se a tecla 'Escape' for pressionada
        msg = 'esc'  # muda a variável msg para 'esc' e força a saída do loop while
        break

    # se o estado principal da FSM = 'BEGIN', crie um socket e tente se conectar ao EnviSim
    while sttMM == Stt.BEGIN:
        # print('<< begin >>')
        try:
            sock.connect(server_address)  # tenta conetar com o servidor na porta definida
            print('Conectado ao Servidor: %s >> porta: %s' % server_address)
            sttMM = Stt.RECEIVING  # depois de conectar, mude a FSM para o estado RECEIVING
        except socket.timeout:  # exceção - tempo esgotado
            # print('ERRO: tempo limite do socket')
            strCode = 'tempo_limite_socket'
            sttMM = Stt.ERRORS  # colocar a máquina no estado ERROS
        except socket.error as e:  # exceção - erro ao se conectar ao servidor
            # print('ERRO ao se conectar ao servidor: %s' % e)
            strCode = 'conexao_servidor'
            sttMM = Stt.ERRORS  # colocar a máquina no estado ERROS
        break

    # se o estado da FSM principal for 'RESTARTING', este programa envia uma solicitação de reinício (RESTARTING) para o EnviSim
    while sttMM == Stt.RESTARTING:
        # print('<< restarting >>')
        msg = '{\"' + keyMagREQ + '\":[\"' + REQrst + '\",0]}'  # solicita ao EnviSim que reinicie a missão
        sttSUBfsm = SubStt.RES  # após o reinício, coloque sempre o subFSM no estado BEGIN
        sttMM = Stt.SENDING  # envia esta mensagem para o EnviSim
        break

    # quando a FSM vai para o estado 'INTERPRETING', o programa deve analisar a mensagem Json recebida
    # aqui convertemos a mensagem do EnviSim em 'spikes' nos neurônios de entrada do SNN
    while sttMM == Stt.INTERPRETING:
        # print('<< interpreting >>')
        # chama o método que interpreta a mensagem
        # ele retorna um novo estado da FSM principal, um código de str (ou ''), e o idx do sensor detectado
        sttMM, strCode, idxInpSensor, CurrentSensBits = interpreting(answES)
        # print('veio:', CurrentSensBits)
    # O FSM entra no estado DECIDING após ter recebido e interpretado uma mensagem
    # Aqui, a "mente" do agente toma decisões e gera: comando/ação/requisição

    while sttMM == Stt.DECIDING:
        # print('<< decidindo >> ', (energy - iterNum))

        if modoDados == 1:  # se estiver no modo de coleta de dados, vai para Stt.WANDERING
            sttMM = Stt.WANDERING
            break

        if iterNum >= energy:  # teste: 'jogo acabou'? O agente não completou a missão
            print('O agente não tem mais ENERGIA!')  # se o jogo acabar, imprime essa mensagem
            strCode = 'semEnergia'  # o agente não tem mais energia - MORREU...
            sttMM = Stt.EXCEPTIONS  # muda o estado para EXCEPTIONS
            break

        # ATENÇÃO: isso está dentro da MAIN STATE-MACHINE, mas é uma FSM secundária
        # esta sub-FSM controla quantas vezes o agente solicita informações do EnviSim
        while sttSUBfsm == SubStt.RES:  # após reiniciar a cena, a subFSM sempre começa a solicitar dados
            if InpSensors[idxInpSensor] == 'inp_' + OUTrst:  # se retornar 'inp_restarted'
                nofIter = 0  # redefinir o número de iterações (energia resetada)
                iterNum = 0
                sttSUBfsm = SubStt.START  # muda o estado desta subFSM para 'solicitar' informações do EnviSim
            else:
                strCode = 'erro => esperava reiniciado...'  # erro não reiniciado
                sttMM = Stt.ERRORS  # muda o estado do FSM principal para EXCEPTIONS
            break

        while sttSUBfsm == SubStt.START:  # for START requesting information only
            cntNofReqs = 0  # set counter of input requirements to 0
            sensInpBits = np.zeros((nofInfoRequest, 32), dtype=np.int32)  # array with nofInfoRequestx32 bits
            sttSUBfsm = SubStt.ASK  # change the state of this subFSM to 'ASK for' info from EnviSim
            break

        while sttSUBfsm == SubStt.ASK:  # permanece no estado ASK enquanto solicita informações
            if cntNofReqs < nofInfoRequest:
                d = str(InfoReqSeq[cntNofReqs][1])  # distância da posição atual que o agente deseja informações
                msg = '{\"' + keyMagREQ + '\":[\"'  # inicia a string de mensagem de solicitação (igual para todas as solicitações)
                if InfoReqSeq[cntNofReqs][0] == 'fwd':  # está solicitando informações de avanço?
                    msg = msg + REQfwd + '\",' + d + ']}'  # completa a string de mensagem de solicitação para avançar
                elif InfoReqSeq[cntNofReqs][0] == 'r90':  # está solicitando informações de 90 graus à direita?
                    msg = msg + REQrgt + '\",' + d + ']}'  # completa a string de mensagem de solicitação para a direita
                elif InfoReqSeq[cntNofReqs][0] == 'l90':  # está solicitando informações de 90 graus à esquerda?
                    msg = msg + REQlft + '\",' + d + ']}'  # completa a string de mensagem de solicitação para a esquerda
                elif InfoReqSeq[cntNofReqs][0] == 'r45':  # está solicitando informações de 45 graus à direita?
                    msg = msg + REQr45 + '\",' + d + ']}'  # completa a string de mensagem de solicitação para a direita
                elif InfoReqSeq[cntNofReqs][0] == 'l45':  # está solicitando informações de 45 graus à esquerda?
                    msg = msg + REQl45 + '\",' + d + ']}'  # completa a string de mensagem de solicitação para a esquerda
                sttMM = Stt.SENDING  # altera o estado para enviar a mensagem
                sttSUBfsm = SubStt.WAITRQ  # altera para o estado WAITRQ (aguardar respostas ao solicitar)
                break
            break

        while sttSUBfsm == SubStt.SAVE:  # estado de SALVAR apenas para salvar a resposta do pedido anterior
            sensInpBits[cntNofReqs] = CurrentSensBits  # salvar as respostas nesta matriz
            cntNofReqs = cntNofReqs + 1  # incrementar o contador (quantos pedidos foram feitos)
            if cntNofReqs == nofInfoRequest:
                sttSUBfsm = SubStt.CMD  # alterar o estado do subFSM para 'enviar' comandos para EnviSim
            else:
                sttSUBfsm = SubStt.ASK  # mudar novamente para o estado ASK até que todos os pedidos sejam feitos
            # print('saving: ', sensInpBits)
            break

        while sttSUBfsm == SubStt.CMD:  # depois de adquirir info, tomar uma decisão e enviar COMANDO p/ EnviSim
            cntNofReqs = 0  # faz cntNofReqs = zero p/ prox interação com EnviSim
            decision = infer(sensInpBits)  # fazer uma inferência (escolher uma ação/saída para ser disparada)
            msg = create_msg(decision, 1)  # converte a decision em uma mensagem p/ EnviSim
            sttMM = Stt.SENDING  # muda o estado da FSM principal p/ SENDING
            sttSUBfsm = SubStt.WAITCM  # muda o sub-estado da subFSM p/ WAITCM (wait for answers after a command)
            break

        while sttSUBfsm == SubStt.CNT:
            fdbkcode = feedback_analysis(sensInpBits, carryRWD)  # faz uma inferência (o que foi recebido como feedback)
            if fdbkcode == -1:
                strCode = 'erro => reiniciando...'  # erro não reiniciável
                sttMM = Stt.ERRORS  # muda o estado da máquina principal para EXCEPTIONS
            elif fdbkcode == 50:  # código para quando o agente pega o ouro
                print('-> a RECOMPENSA foi coletada <-')  # o agente completou a missão com sucesso
                carryRWD = 1
                nofIter = 0  # restaura a energia do agente
            elif fdbkcode == 100:  # código para sucesso!
                print('O Agente GANHOU - sucesso!')  # o agente completou a missão com sucesso
                strCode = OUTsuc  # string para sucesso
                sttMM = Stt.EXCEPTIONS  # muda o estado para EXCEPTIONS
                sttSUBfsm = SubStt.ASK  # muda para o estado ASK
                break
            elif fdbkcode == -100:  # código para morte!
                print('O Agente MORREU ')  # o agente morreu
                strCode = OUTdie  # string para morte
                sttMM = Stt.EXCEPTIONS  # muda o estado para EXCEPTIONS
                sttSUBfsm = SubStt.ASK  # muda para o estado ASK
                break
            msg = create_msg(fdbkcode, 0)  # transforma o feedback em uma mensagem
            sttMM = Stt.SENDING  # muda o estado da máquina para SENDING
            sttSUBfsm = SubStt.ASK  # muda para o estado ASK
            break

        while sttSUBfsm == SubStt.WAITRQ:
            sttSUBfsm = SubStt.SAVE  # mudar para o estado SAVE para salvar as respostas
            break

        while sttSUBfsm == SubStt.WAITCM:
            # este atraso é apenas para efeitos visuais - remova-o para simulações mais rápidas
            sttSUBfsm = SubStt.CNT  # mudar para o estado CNT
            break


    # quando a FSM está no estado EXCEPTIONS, um comando/mensagem EnviSim para este programa foi recebido
    # hierarquicamente, essas mensagens são mais importantes do que as 'SENSE'
    # eles são: não pode, morreu, pegou, nenhum, sucesso (eles significam o resultado de uma ação)
    # você deve decidir: o que fazer quando o agente morrer? ou pegar o ouro? ou ter sucesso? etc.
    # deveria 'SALVAR' uma missão? deveria simplesmente 'REINICIAR' EnviSim? (é aberto)
    while sttMM == Stt.EXCEPTIONS:
        # print('EXCEPTION - dealing with a special msg')  # apenas imprime uma nota para esta situação

        if modoDados == 1:  # se estiver no modo de coleta de dados, vai para Stt.WANDERING
            sttMM = Stt.WANDERING
            break

        if strCode == OUTrst:  # recebeu 'restarted' - primeira coisa a fazer: pedir informações
            msg = '{\"' + keyMagREQ + '\":[\"' + REQfwd + '\",1]}'  # solicita informações sobre 1 posição à frente
            sttMM = Stt.SENDING  # muda o estado da máquina para SENDING
            break

        elif strCode == keyMwpPOS:  # trate o que você fará com posX e posY
            posX = posX  # o valor atual de x - você irá salvá-lo?
            posY = posY  # o valor atual de y - você irá salvá-lo?
            msg = '{\"' + keyMagREQ + '\":[\"' + REQfwd + '\",1]}'  # solicita informações ???
            sttMM = Stt.SENDING  # muda o estado da máquina para SENDING
            break

        elif strCode == 'noEnergy':  # trate o caso quando o agente morreu sem energia
            # você irá salvar a missão atual?
            msg = '{\"' + keyMagREQ + '\":[\"' + REQrst + '\",0]}'  # solicita um reset imediato ???
            nofIter = 0  # reinicia o valor para o número de iterações
            sttMM = Stt.SENDING  # muda o estado da máquina para SENDING
            break

        elif strCode == OUTdie:  # trate o caso quando o agente morreu!!!
            # você irá salvar a missão atual?
            msg = '{\"' + keyMagREQ + '\":[\"' + REQrst + '\",0]}'  # solicita que EnviSim reinicie a missão
            nofIter = 0  # reinicia o valor para o número de iterações
            sttMM = Stt.SENDING  # muda o estado da máquina para SENDING
            break

        elif strCode == OUTsuc:  # Ótimo!!! Parece que o agente alcançou com sucesso o final da missão
            msg = '{\"' + keyMagREQ + '\":[\"' + REQrst + '\",0]}'  # solicita um reset imediato ???
            nofIter = 0  # reinicia o valor para o número de iterações
            carryRWD = 0  # restaura a condição de não ter recebido a recompensa

            # o que fazer com a missão atual?

            sttMM = Stt.SENDING  # altera o estado da máquina para SENDING
            break
        sys.exit(-2)

    # este é um estado para testes, que pode ser excluído na versão final deste programa
    while sttMM == Stt.ERRORS:
        print('--> estado ERRORS::')
        print(strCode)
        sys.exit(-1)
        # você pode colocar aqui qualquer tipo de código para tratar erros

    # se o estado principal da FSM for 'RECEIVING', aguarda uma resposta do EnviSim
    while sttMM == Stt.RECEIVING:
        # print('<< receiving >>')
        try:
            answES = sock.recv(256)  # recebe uma mensagem com até 256 caracteres
            #print('resposta_conn: %s' % answES)
            sttMM = Stt.INTERPRETING  # resposta recebida, altera o estado para INTERPRETING
        except socket.error as e:  # se ocorrer algum erro, imprime o erro do socket
            print('Erro de Socket: ', str(e))
            strCode = 'socket_error'
            sttMM = Stt.ERRORS  # coloca a máquina no estado ERRORS
        break

    # se o estado da FSM for 'SENDING', este programa envia o conteúdo da variável 'msg' para o EnviSim
    while sttMM == Stt.SENDING:
        # print('<< sending >>')
        #print('enviando = ', msg)
        if msg != '':  # testa se a variável 'msg' não está vazia
            try:
                sock.sendall(msg.encode('utf-8'))  # envia a mensagem via socket
                # print('sent: ', msg)  # imprime a mensagem enviada para o EnviSim
                msg = ''  # limpa a string na variável 'msg'
                sttMM = Stt.RECEIVING  # altera a FSM para o estado RECEIVING
            except socket.error as e:  # se ocorrer algum erro, imprime o erro do socket
                print('Erro de Socket: ', str(e))
                strCode = 'socket_error'
                sttMM = Stt.ERRORS  # coloca a máquina no estado ERRORS
            break
        else:  # erro - tentando enviar uma mensagem vazia
            print('Atenção: tentando enviar uma mensagem vazia')
            strCode = 'empty_msg'
            sttMM = Stt.ERRORS  # altera para o estado que trata os ERROS
            break

    # estado para testes - pode ser deletado no final...
    while sttMM == Stt.WANDERING:
        print('>> state WANDERING << ', nofWandSteps)
        # este atraso é apenas para efeitos visuais - remova-o para simulações mais rápidas

        if subSttWand == 0:  # substado zero, pedir a posição atual no grid do EnviSim
            rscSeqWand = [0, -1, -1, -1, -1]  # 0=não tem ouro, cria uma lista para rascunho
            # criar msg = request for distância = 0
            msg = create_msg(4, 0)  # (4,0) porque 4='req_forward' e 0=distância
            # prox estado da mainFSM => enviar (envia, recebe, interpreta, decide ==> Stt.WANDERING)
            subSttWand = 1  # prox substado será o abaixo
            sttMM = Stt.SENDING  # prox estado é enviar a mensagem
        elif subSttWand == 1:  # faz uma escolha randômica de um comando
            rscSeqWand[1] = idxInpSensor  # rscSeqWand[0] coloca 'onde está' no grid
            # sorteia um dos comandos, restritos, a ser enviado para o EnviSim
            idx_rest = [0, 1, 3, 4, 5, 9, 11, 12, 13]  # indxs restritos
            idx_rand = random.choice(idx_rest)  # escolhe um desses indxs
            rscSeqWand[2] = idx_rand  # em rscSeqWand[1] coloca o 'comando' escolhido (idx_rand)
            # criar msg = idx_rand e dist=1
            msg = create_msg(idx_rand, 1)  # idx_rand=comando sorteado, 1=distância
            # prox estado da mainFSM => enviar (envia, recebe, interpreta, decide ==> Stt.WANDERING)
            subSttWand = 2  # prox substado será o abaixo
            sttMM = Stt.SENDING  # prox estado é enviar a mensagem
        elif subSttWand == 2:  # veio com a resposta do que acontece depois do comando
            rscSeqWand[3] = idxInpSensor  # em rscSeqWand[2] coloca o 'resultado'

            # esse if decide se o label será bom, ruim ou neutro
            if idxInpSensor in [0, 18]:  # nothing, none
                rscSeqWand[4] = 0
            elif idxInpSensor in [3, 8, 9, 11]:  # qq um que tenha flash
                rscSeqWand[4] = 2
            elif idxInpSensor in [1, 7, 10]:  # qq um que tenha breeze ou stench
                rscSeqWand[4] = -1
            elif idxInpSensor in [6, 12, 13, 14]:  # qq um que tenha obstáculo
                rscSeqWand[4] = -2
            elif idxInpSensor in [2, 16]:  # qq um que tenha morte é mau
                rscSeqWand[4] = -5
            elif idxInpSensor in [4, 5]:  # encontrou goal ou o initial
                rscSeqWand[4] = 2.5
            elif idxInpSensor in [15]:  # encontrou cannot *
                rscSeqWand[4] = -2
            elif idxInpSensor in [17]:  # veio grabbed *
                rscSeqWand[0] = 1
                rscSeqWand[4] = 6
            elif idxInpSensor in [19]:  # veio restarted *
                rscSeqWand[4] = 6
            elif idxInpSensor in [20]:  # veio success *
                rscSeqWand[4] = 10
            seqWand.append(rscSeqWand)  # insere a sequência rascunho na lista final
            rasc = random.randint(1, 12)

            if rasc > 4:
                msg = create_msg(3, 1)  # (3,1) porque 3='MOV_forward' e 1=dist (sempre anda +1)
            elif rasc > 2:
                msg = create_msg(12, 2)  # (12,2) porque 12='ROT_right' e 2=90 (gira)
            else:
                msg = create_msg(11, 2)  # (11,2) porque 11='ROT_left' e 2=90 (gira)
            # prox estado da mainFSM => enviar (envia, recebe, interpreta, decide ==> Stt.WANDERING)
            subSttWand = 3  # prox substado será o abaixo
            sttMM = Stt.SENDING  # prox estado é enviar a mensagem
        elif subSttWand == 3:  # andou uma casa no grid (ou morreu, colidiu, etc)
            if nofWandSteps > 0:
                nofWandSteps = nofWandSteps-1  # decrementa o número de steps coletando dados
                subSttWand = 0  # volta o substado para zero
                sttMM = Stt.WANDERING  # prox estado é este mesmo (WANDERING)
            else:  # fim do vaguear, nofWandSteps==0, salvar o arquivo
                with open(arqv_csv, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(seqWand)
                    print('Sessão gravada com sucesso')
                    msg = 'esc'
                    sttMM = Stt.EXCEPTIONS
        break

else:  # se msg == 'esc' esse processo (programa) será fechado
    sock.close()  # close the socket
    print('<< END of process >>')  # imprime a mensagem de finalização do processo

sys.exit(0)
# ----(end)----- MAIN = end of main program  --------

