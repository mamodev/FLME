<div class="titlepage">

![image](cherubinFrontespizio.eps)

<div class="center">

UNIVERSITÀ DI PISA  
Laurea Triennale in Informatica

</div>

<div class="center">

**A user-friendly framework for cross-device asynchronous Federated
Learning**

</div>

Relatore: **  
Prof: Massimo Torquati **  
Prof: Patrizo Dazzi****

Candidato: **  
Marco Morozzi**

  

</div>

# Introduzione

Si stima che nel mondo siano in uso oltre 4.6[^1] miliardi di smartphone
e più di 30[^2] miliardi di dispositivi IoT connessi a livello globale,
un numero in costante crescita . I dispositivi IoT e mobile generano
quotidianamente una mole straordinaria di dati, dai messaggi di testo ai
parametri biometrici, dalle transazioni online alle abitudini di
consumo, fino ai dati raccolti da sensori ambientali e wearable. Secondo
l’International Data Corporation (IDC), il volume di dati generato
dall’IoT a livello globale potrebbe raggiungere 180 zettabyte (ZB) entro
il 20 . Tali informazioni rappresentano un’enorme opportunità per il
Machine Learning e per lo sviluppo di modelli predittivi in grado di
trasformare i dati grezzi in conoscenza. Tuttavia, il paradigma
tradizionale dell’apprendimento automatico richiede che i dati vengano
raccolti, centralizzati ed elaborati in data center dedicati o su server
cloud. Sebbene questo approccio sia stato ampiamente utilizzato,
presenta criticità sempre più evidenti sia in termini di scalabilità che
di rispetto della privacy.

#### Le sfide della centralizzazione dei dati

La centralizzazione dei dati comporta due grandi problematiche. La prima
riguarda la dimensione stessa dei dati, che richiede risorse di rete
significative per il loro trasferimento. Con miliardi di dispositivi che
generano dati in tempo reale, le infrastrutture necessarie per
supportare questo flusso continuo di informazioni diventano estremamente
costose sia da implementare che da mantenere. Inoltre, l’archiviazione
centralizzata di tali volumi di dati, che richiede infrastrutture
fisiche ed energetiche non sostenibile su larga scala, aggrava il
problema dell’impatto ambientale ed aumenta significativamente i costi
operativi per garantire la continuità dei servizi.

La seconda problematica, forse ancora più critica della prima, riguarda
la privacy e la sicurezza dei dati. Trasferire grandi quantità di dati
sensibili, come informazioni personali, dati sanitari o dettagli
finanziari, verso un’unica entità centrale introduce rischi
significativi. Violazioni della privacy, accessi non autorizzati,
cyberattacchi e utilizzi impropri dei dati sono alcune delle principali
minacce legate a questo approccio. Queste preoccupazioni sono accentuate
dalla crescente attenzione delle normative internazionali, come il
*Regolamento Generale sulla Protezione dei Dati* (denominato GDPR) in
Europa , che impongono standard sempre più rigidi per la gestione e la
protezione dei dati personali.

#### Un nuovo paradigma: il Federated Learning

In risposta a queste sfide, il Federated Learning (FL) emerge come un
paradigma innovativo che permette di superare i limiti del Machine
Learning centralizzato. Questo approccio consente di spostare il calcolo
verso i dati, anziché trasferire i dati verso il calcolo. In altre
parole, i modelli vengono addestrati direttamente sui dispositivi
locali, come smartphone, tablet o sensori IoT, utilizzando i dati
presenti e raccolti sul dispositivo stesso. Solo i parametri aggiornati
del modello vengono inviati a un server centrale per l’aggregazione,
preservando così la privacy e riducendo drasticamente la quantità di
dati trasmessi.

Il Federated Learning offre una risposta efficace alle problematiche di
privacy e scalabilità. La possibilità di addestrare modelli direttamente
sui dispositivi apre nuove prospettive per il learning automatico,
specialmente in settori critici come la sanità, la finanza e le
applicazioni consumer. Ad esempio, il completamento automatico delle
parole sulla tastiera di smartphone (ad esempio Gboard di Google ) o i
sistemi di riconoscimento vocale negli assistenti virtuali (ad esempio
Apple Siri ) sono già oggi esempi concreti di Federated Learning
applicato su larga scala.

#### Le sfide del Federated Learning

Nonostante i vantaggi, implementare un sistema di Federated Learning
efficiente, scalabile e sostenibile in contesti cross-device, ossia su
reti altamente distribuite e composte da un numero elevato di
dispositivi eterogenei, non è un compito banale. I dispositivi che
partecipano a questi sistemi variano notevolmente in termini di risorse
computazionali, capacità di memoria e connettività. Inoltre, i dati
raccolti da questi dispositivi sono spesso non indipendenti e non
identicamente distribuiti (non-i.i.d.), riflettendo le caratteristiche
uniche degli utenti o dei contesti in cui vengono generati. Questa
eterogeneità statistica complica la creazione di un modello globale che
sia rappresentativo e performante per tutti i partecipanti .

Anche dal punto di vista infrastrutturale, il Federated Learning pone
sfide significative. La gestione di milioni di dispositivi implica la
necessità di protocolli di comunicazione robusti e di algoritmi di
aggregazione che possano operare in modo efficace nonostante la
partecipazione dinamica e imprevedibile dei dispositivi, molti dei quali
potrebbero disconnettersi durante l’addestramento. Questi aspetti
rendono cruciale la progettazione di sistemi resilienti, capaci di
tollerare latenze elevate nelle comunicazioni, nonché errori di
comunicazione dei dati ed una partecipazione altamente dinamica delle
entità della rete di comunicazione.

#### La motivazione e gli obiettivi della tesi

Mentre il Federated Learning si afferma come una tecnologia promettente,
l’implementazione di soluzioni pratiche che scalino a reti cross-device
richiede competenze tecniche avanzate e trasversali, spesso lontane dal
focus principale di un esperto di Machine Learning. La progettazione di
un sistema di Federated Learning implica infatti la risoluzione di
problemi legati all’architettura del sistema, alla gestione delle
comunicazioni, alla sincronizzazione dei dispositivi e
all’ottimizzazione degli algoritmi di aggregazione. Questo può
rappresentare un ostacolo significativo per i ML Model Developers e Data
Scientist il cui obiettivo principale in questo contesto è progettare e
addestrare modelli per casi d’uso specifici.

Questa tesi nasce proprio da queste considerazioni. L’obiettivo è
sviluppare un framework user-friendly che renda l’adozione del Federated
Learning accessibile anche agli sviluppatori non esperti in
infrastrutture distribuite. Il framework proposto mira a semplificare il
processo di implementazione, fornendo strumenti che permettano agli
sviluppatori di concentrarsi prevalentemente sulla definizione del
modello e sulla raccolta dei dati, senza doversi preoccupare delle
complessità tecniche sottostanti.

Inoltre, il framework è progettato per affrontare le sfide specifiche
del contesto cross-device, integrando soluzioni per la gestione
dell’eterogeneità dei dispositivi, la variabilità statistica dei dati e
le limitazioni delle risorse computazionali. L’obiettivo finale è
offrire un ambiente di sviluppo che sia trasparente per l’utente,
replicando l’esperienza di un sistema centralizzato ma beneficiando
delle caratteristiche distribuite e di privacy-preserving del Federated
Learning.

#### Struttura della tesi

Nei capitoli successivi, verranno introdotti i principi fondamentali del
Machine Learning e del Federated Learning, con un focus particolare
sulle sfide che caratterizzano gli scenari cross-device. Saranno
analizzati i principali approcci per gestire l’eterogeneità dei
dispositivi e dei dati, così come le architetture software che
supportano l’apprendimento federato. In seguito verranno discusse le
soluzioni attualmente presenti sul mercato ed introdotta la struttura
del framework elaborato oggetto di questa tesi descrivendone le
caratteristiche tecniche, le innovazioni introdotte e i risultati
sperimentali ottenuti, evidenziando il suo contributo nel rendere il
Federated Learning più accessibile, scalabile ed efficace.

#### Codice del framework

Il codice sorgente del framework disegnato e sviluppato in questa tesi è
pubblico e scaricabile da Github al seguente link:  
<https://github.com/mamodev/Async-Federated-Learnig>

# Background

In questo capitolo vengono presentate le informazioni e le conoscenze di
base necessarie per comprendere il contesto del Federated Learning, con
particolare attenzione al suo utilizzo in ambienti cross-device e alle
sfide che esso comporta. Verranno introdotti i principi fondamentali
dell’apprendimento federato, un paradigma di Machine Learning progettato
per preservare la privacy dei dati distribuendo il processo di
addestramento su diversi dispositivi. Saranno inoltre esplorati i
concetti chiave di eterogeneità statistica, comunicazione asincrona e
gestione dei Client, essenziali per comprendere le complessità di un
sistema di Federated Learning cross-device.

Questa panoramica fornisce al lettore una solida base per apprezzare il
contributo tecnico e metodologico del lavoro svolto in questa tesi, che
mira a sviluppare un framework user-friendly per implementare algoritmi
di Federated Learning asincroni in contesti cross-device, affrontando le
sfide di scalabilità, efficienza e semplicità d’uso per l’utente.

## Machine Learning

Il Machine Learning (ML) è una disciplina dell’intelligenza artificiale
che si concentra sullo sviluppo di algoritmi e modelli statistici capaci
di apprendere e migliorare le proprie prestazioni attraverso
l’esperienza, senza essere esplicitamente programmati per svolgere un
compito specifico. In pratica, il ML consente ai computer di
identificare pattern e inferire regole dai dati, permettendo loro di
fare previsioni o prendere decisioni basate su nuovi input.  
  
I modelli di Machine Learning vengono addestrati utilizzando insiemi di
dati, che possono contenere esempi etichettati o non etichettati. A
seconda del tipo di apprendimento e della natura dei dati, il ML si
suddivide principalmente in tre categorie:

-   **Apprendimento Supervisionato**: Il modello viene addestrato su un
    dataset etichettato, dove ogni esempio di input è associato a un
    output desiderato. L’obiettivo è apprendere una funzione che mappa
    gli input agli output corretti, permettendo al modello di fare
    previsioni accurate su dati nuovi e non visti.

-   **Apprendimento Non Supervisionato**: Il modello lavora con dati non
    etichettati e cerca di identificare strutture nascoste o pattern
    all’interno dei dati. Questo tipo di apprendimento è utilizzato per
    compiti come il clustering, dove gli esempi vengono raggruppati in
    base a somiglianze intrinseche.

-   **Apprendimento per Rinforzo**: Il modello, spesso chiamato agente,
    interagisce con un ambiente dinamico e apprende a compiere azioni
    che massimizzano una ricompensa cumulativa. L’agente prende
    decisioni sequenziali, adattandosi in base al feedback ricevuto
    dalle sue azioni precedenti.

Per lo scopo di questa tesi ci concentriamo su ML supervisionato il
quale può essere descritto attraverso le seguenti fasi:

-   **Raccolta dei Dati**: I dati vengono acquisiti da diverse fonti,
    come dispositivi mobili, sensori, applicazioni web o database
    aziendali. Questi dati possono includere informazioni strutturate o
    non strutturate, e spesso contengono dati sensibili o personali.

-   **Trasferimento e Centralizzazione dei Dati**: I dati raccolti
    vengono trasferiti attraverso reti di comunicazione al server
    centrale. Questo processo può comportare l’invio di grandi volumi di
    dati, richiedendo una larghezza di banda significativa e
    introducendo potenziali rischi per la sicurezza durante il
    trasferimento.

-   **Addestramento del Modello**: Il modello di Machine Learning viene
    addestrato utilizzando i dati pre-elaborati. Questo coinvolge l’uso
    di algoritmi di ottimizzazione per minimizzare una funzione di
    perdita, adattando i parametri del modello in modo da migliorare la
    sua capacità di fare previsioni accurate.

-   **Validazione e Test**: Il modello addestrato viene valutato su un
    set di dati di validazione per misurare le sue prestazioni e
    prevenire problemi come l’overfitting. Se necessario, si iterano
    processi di tuning degli iperparametri o si esplorano architetture
    alternative per migliorare i risultati.

-   **Implementazione e Distribuzione**: Una volta che il modello
    soddisfa i criteri di prestazione desiderati, viene implementato in
    un ambiente di produzione. Questo può includere l’integrazione in
    applicazioni software, servizi web o sistemi embedded, dove il
    modello fornisce previsioni o prende decisioni in tempo reale.

Il Machine Learning ha rivoluzionato numerosi settori, tra cui il
riconoscimento vocale, l’elaborazione del linguaggio naturale, la
visione artificiale e l’analisi predittiva. Tuttavia, l’approccio
tradizionale di addestramento prevede la raccolta e la centralizzazione
dei dati provenienti da varie fonti, un approccio che che può comportare
rischi significativi in termini di privacy, sicurezza e conformità
normativa. Questi rischi emergono dal trasferimento e dall’archiviazione
di dati potenzialmente sensibili in un unico luogo, esponendoli a una
maggiore vulnerabilità rispetto ad accessi non autorizzati e a possibili
violazioni della privacy.

## Federated Learning

Il Federated Learning (FL) è una tecnica innovativa di machine learning
supervisionato che consente l’addestramento di modelli di intelligenza
artificiale su dati che rimangono distribuiti su diverse fonti, senza la
necessità di centralizzare i dati in un unico punto. Esso inverte il
paradigma classico del machine learning centralizzato spostando
l’addestramento dei modelli dove si trovano i dati invece di spostare i
dati dove avviene l’addestramento dei modelli. Nel machine learning
centralizzato, i dati provenienti da diverse fonti vengono raccolti e
trasferiti a un server centrale o a un data center. Qui, i dati vengono
elaborati e utilizzati per addestrare il modello di machine learning.Nel
Federated Learning (Apprendimento Federato), il calcolo viene spostato
verso i dati. Invece di trasferire i dati a un server centrale,
l’addestramento del modello avviene localmente sui dispositivi che
possiedono i dati (esempi di dispositivi sono smarphone, desktop e
sensori con capacità di calcolo). Solo i parametri del modello
aggiornati vengono inviati a un server centrale, dove vengono aggregati
per aggiornare il modello globale, garantendo che i dati personali non
vengano mai esposti o trasferiti.

L’apprendimento federato è particolarmente importante in contesti dove
la privacy e la sicurezza dei dati sono cruciali, come nel settore
sanitario o finanziario. Inoltre, consente di sfruttare la potenza di
calcolo distribuita, riducendo la necessità di infrastrutture
centralizzate costose e complesse, nonché ridurre i requisiti di banda
necessari per la trasmissione di grandi dataset in un unico centro di
elaborazione.

In modo esemplificativo possiamo affermare che:

-   Il Machine Learning centralizzato prevede lo spostamento dei dati
    verso il calcolo

-   Il Federated (Machine) Learning prevede lo spostamento del calcolo
    verso i dati.

<figure>
<img src="fed_overview" id="fig:architetturaFL" style="width:14cm"
alt="Architettura logica dell’apprendimento federato. I modelli indicati con Model 1,...,Model N sono addestrati localmente su dispositivi differenti ed accedendo esclusivamente a dati locali accessibili al dispositivo. I dati ottenuti dall’addestramento dei modelli locali (ad esempio parametri o pesi) vengono inviati ad un modello aggregato centrale (Aggregated Model) nel Cloud che potrà essere ritrasmesso come aggiornamento del modello locale ai dispositivi. " />
<figcaption aria-hidden="true">Architettura logica dell’apprendimento
federato. I modelli indicati con <em>Model 1</em>,...,<em>Model N</em>
sono addestrati localmente su dispositivi differenti ed accedendo
esclusivamente a dati locali accessibili al dispositivo. I dati ottenuti
dall’addestramento dei modelli locali (ad esempio parametri o pesi)
vengono inviati ad un modello aggregato centrale (Aggregated Model) nel
Cloud che potrà essere ritrasmesso come aggiornamento del modello locale
ai dispositivi. </figcaption>
</figure>

In Figura <a href="#fig:architetturaFL" data-reference-type="ref"
data-reference="fig:architetturaFL">2.1</a> è schematizzata
l’architettura logica dell’approccio all’apprendimento federato. Questo
modello architetturale è stato introdotto da Google nel 2016 per
preservare la privacy degli utenti e ridurre la necessità di
trasferimento di grandi quantità di dati sensibili .

#### Cross-Silo vs Cross-Device

Il termine Federated Learning è stato inizialmente introdotto con
un’enfasi particolare sulle applicazioni per dispositivi mobili ed Edge,
quindi a larghissima scala. Il termine “Edge” si riferisce a dispositivi
o nodi di calcolo posizionati ai margini della rete, vicini alle fonti
dei dati o agli utenti finali. Questi includono smartphone, sensori IoT
e piccoli server locali che eseguono elaborazioni direttamente sul
posto, riducendo così la latenza e il carico sulla rete centrale. A
differenza dell’architettura Cloud tradizionale, dove i dati vengono
inviati a data center centralizzati per l’elaborazione e
l’archiviazione, l’architettura Edge sposta parte del carico
computazionale verso i dispositivi periferici. Questo permette
un’elaborazione più rapida e un minore utilizzo della larghezza di
banda, poiché i dati non devono essere trasmessi interamente al Cloud.

Tuttavia, l’interesse per l’applicazione del Federated Learning si è
notevolmente ampliato, includendo anche scenari che coinvolgono un
numero limitato di Clienti relativamente affidabili. Un esempio di
questi scenari è la collaborazione tra più organizzazioni distinte che
vogliono cooperare per addestrare un modello senza però condividere i
dati di ciascuna organizzazione. Queste due distinte impostazioni di
Federated Learning hanno dato vita a due modelli architetturali distinti
denominati *Cross-Silo* e *Cross-Device*.

##### Cross-Silo Federated Learning.

Questo modello architetturale di FL coinvolge organizzazioni, aziende o
gruppi di clienti. In questo scenario, il numero di partecipanti è
solitamente ridotto (ad esempio, qualche centinaio), ma ogni entità
locale possiede una quantità significativa di dati locali. Un esempio
pratico di Cross-Silo Federated Learning è la collaborazione tra
istituti medici per addestrare modelli di previsione delle malattie
utilizzando dati sensibili dei pazienti di ogni centro. In questo
contesto in in cui il numero di partecipanti nell’addestramento del
modello è relativamente piccolo e tipicamente corrisponde ad
organizzative come aziende o istituzioni, ciascuna ha a disposizione
grandi volumi di dati e grandi risorse computazionali affidabili. In
questo scenario, la partecipazione delle entità all’apprendimento
federato è più stabile e coordinata, grazie anche a connessioni di rete
affidabili e ad alta velocità.

##### Cross-Device Federated Learning.

Questo modello coinvolge dispositivi mobili come smartphone, wearable
sensors e dispositivi IoT. In questo caso, il numero di Client può
raggiungere valori estremamente grandi (centinaia di migliaia o
milioni), ma ogni Client ha una quantità relativamente piccola di dati
locali. Questo approccio richiede la partecipazione di un gran numero di
dispositivi edge per avere successo. Questa configurazione presenta
alcune sfide peculiari, come la partecipazione dinamica e non
prevedibile dei dispositivi, le risorse limitate (in termini di capacità
computazionale, memoria e batteria), e la variabilità delle connessioni
che possono essere instabili.

In sintesi, il Cross-Silo Federated Learning è adatto per scenari in cui
poche organizzazioni collaborano utilizzando grandi quantità di dati e
risorse di calcolo significative, mentre il Cross-Device Federated
Learning è tipico di scenari in cui molti (o moltissimi) dispositivi con
piccole quantità di dati e potenza di calcolo relativa collaborano per
addestrare un modello globale.

Per lo scopo di questo tesi ci limiteremo ad analizzare principalmente
il contesto cross-device che presenta sfide ed opportunità maggiori.

#### Esempi di Applicazione dell’Apprendimento Federato

Il Federated Learning ha trovato applicazione in diversi ambiti,
dimostrando la sua efficacia nel preservare la privacy dei dati e nel
migliorare l’efficienza dei modelli su dispositivi decentralizzati. Un
esempio rilevante è il completamento automatico della tastiera sui
dispositivi mobili, come implementato da Google nel suo servizio Gboard
. In questo caso, i modelli apprendono dai dati di digitazione degli
utenti direttamente sui loro dispositivi, evitando il trasferimento di
informazioni sensibili verso server centrali e garantendo così la
privacy.

Un altro esempio riguarda gli assistenti virtuali integrati negli
smartphone, che utilizzano il Federated Learning per perfezionare i
sistemi di riconoscimento vocale. Grazie a questo approccio, i
dispositivi possono apprendere dai comandi vocali degli utenti,
migliorando le capacità di comprensione senza trasmettere registrazioni
vocali ai server, riducendo al minimo i rischi per la riservatezza.

In ambito sanitario, i dispositivi indossabili (wearable devices)
raccolgono dati relativi alla salute, come il battito cardiaco, la
pressione arteriosa e l’attività fisica che si sta svolgendo. Attraverso
il Federated Learning, questi dispositivi possono addestrare modelli
predittivi per monitorare le condizioni di salute degli utenti
direttamente sui dispositivi stessi, senza trasferire dati sensibili al
Cloud di riferimento dell’applicazione. In questo modo, si garantisce la
riservatezza delle informazioni personali, preservando la privacy degli
utenti.

In tutte queste applicazioni, è possibile riconoscere un insieme comune
di passaggi necessari per definire il modello di apprendimento federato.
A titolo esemplificativo, i principali passaggi sono elencati di
seguito:

-   **Inizializzazione del modello globale**: Si inizia con
    l’inizializzazione del modello globale (e dei suoi pesi) sul server
    centrale (denominato Master). Il modello può essere inizializzato
    casualmente o basato su un modello precedentemente addestrato.

-   **Selezione dei Client**: Il Master seleziona, con una qualche
    politica, un sottoinsieme delle entità periferiche (denominate
    Client) che devono partecipare ad un round di addestramento.

-   **Fornire il modello globale ai Client**: Vengono inviati ai Client
    i parametri dell’ultima versione del modello, insieme ai parametri
    necessari per la fase di addestramento, come il numero di epoche, la
    dimensione dei batch, il learning rate, ecc.

-   **Allenamento locale**: Ogni Client addestra il modello ricevuto dal
    Master utilizzando i propri dati locali. Al termine
    dell’addestramento, il Client invia al Master il modello locale
    aggiornato ed eventualmente anche alcuni metadati non sensibili dal
    punti di vista della privacy come la dimensione del dataset
    utilizzato.

-   **Aggregazione**: Una volta che ciascun Client selezionato ha
    inviato i risultati dell’addestramento, il Master combina tutti i
    modelli locali per creare un nuovo modello globale. L’aggregazione
    dei modelli locali può seguire diverse strategie e algoritmi; il più
    semplice è chiamato *Federated Averaging* (FedAvg) , che calcola la
    media ponderata dei modelli locali utilizzando come peso la
    dimensione del dataset locale di ciascun Client.

-   **Controllo di convergenza**: Il processo di apprendimento è un
    processo iterativo, che viene ripetuto fino al raggiungimento della
    convergenza del modello globale.

#### Sfide dell’apprendimento federato

Sebbene il Federated Learning presenti numerosi vantaggi rispetto al
Machine Learning centralizzato, esso non è privo di problematiche e
sfide significative che devono essere affrontate e risolte.

L’*eterogeneità dei sistemi e dispositivi* che partecipano alla
definizione del modello aggregato, e l’*eterogeneità statistica* dei
dati introducono problemi che hanno un impatto non trascurabile. Per
**eterogeneità dei sistemi** si intende la variabilità delle
caratteristiche hardware e di connettività tra i dispositivi. Alcuni
dispositivi potrebbero avere risorse molto limitate (e/o limitazione sul
consumo energetico) e quindi potrebbero non essere in grado di sostenere
il training di reti complesse. Per **eterogeneità statistica** si
riferisce alla variazione e alla distribuzione non uniforme dei dati tra
i dispositivi partecipanti. In un contesto distribuito su vari
dispositivi i dati riflettono le attività e le caratteristiche
specifiche degli utenti o dei dispositivi, introducendo differenze
significative nelle statistiche dei dati locali. Questa variabilità
statistica può manifestarsi in vari modi:

-   *Distribuzione dei dati non i.i.d.*: i dati locali su ciascun
    dispositivo non seguono la stessa distribuzione (non sono
    \`̀indipendenti e identicamente distribuiti'́).

-   *Disparità nella quantità di dati*: ogni dispositivo potrebbe avere
    una quantità diversa di dati locali a disposizione per
    l’addestramento.

-   *Differenze nei pattern dei dati*: i dati locali possono contenere
    pattern diversi a seconda del contesto in cui vengono generati sulla
    base di fattori come la regione geografica, le preferenze personali
    e il comportamento.

L’eterogeneità dei sistemi e statistica pone sfide significative per il
Federated Learning, poiché rende difficile creare un modello globale che
funzioni bene per tutti i Client che partecipano all’addestramento.
Algoritmi specifici, come FedAvg , devono essere adattati per gestire
questa variabilità e garantire che l’aggregazione dei modelli locali
porti a un modello globale efficace, pur rispettando le caratteristiche
uniche di ciascun Client. Inoltre possono introdurre aspetti come:

-   **Convergenza rallentata**: la disomogeneità dei dati rallenta la
    convergenza degli algoritmi di Federated Learning.

-   **Incoerenza dell’obiettivo**: l’aggregazione di modelli addestrati
    su dati non i.i.d. può portare alla convergenza verso un punto
    stazionario di una funzione obiettivo diversa da quella reale.

-   **Instabilità del modello**: la differenza tra i modelli locali e
    quello globale, dovuta ai dati non i.i.d., può portare a una
    convergenza instabile che può non convergere mai alla funzione
    obiettivo globale.

## Eterogeneità dei Sistemi di Calcolo

In , è stato introdotto l’algoritmo Federated Averaging (FedAvg), che
permette ai Client di aggiornare i propri modelli locali prima di
inviare i parametri aggiornati al Server centrale per l’aggregazione.
FedAvg prevede che ciascun Client esegua *E* epoche (iterazioni sul
proprio dataset locale) utilizzando lo Stochastic Gradient Descent (SGD)
con una dimensione di mini-batch pari a *B*. Di conseguenza, per un
Client con *n*<sub>*i*</sub> campioni di dati locali, il numero di
iterazioni locali di SGD è calcolato come
*t*<sub>*i*</sub> = *E* ⋅ *n*<sub>*i*</sub>/*B*, valore che può variare
notevolmente tra i Client.

Questa variazione è causata dall’eterogeneità dei sistemi di calcolo, un
aspetto intrinseco del Federated Learning in cui i Client dispongono di
risorse computazionali e dataset locali di dimensioni diverse. In
particolare, dispositivi come smartphone, tablet e laptop differiscono
in termini di capacità di elaborazione, memoria disponibile e durata
della batteria, il che influenza la quantità di lavoro computazionale
che ciascun Client può gestire. Inoltre, anche a parità di risorse di
calcolo, rallentamenti possono essere causati da processi che girano in
background, interruzioni, limitazioni di memoria. In generale, un
dispositivo con un dataset locale più ampio (*n*<sub>*i*</sub> elevato)
richiederà più iterazioni per completare *E* epoche, mentre un
dispositivo con risorse limitate potrebbe incontrare difficoltà anche
solo ad eseguire poche iterazioni tra quelle previste.

L’eterogeneità tra i Client comporta anche una variabilità significativa
nelle prestazioni di addestramento locale, influenzando sia la velocità
di convergenza che la qualità del modello aggregato. Per affrontare
questa sfida, FedAvg e altri algoritmi di Federated Learning devono
essere progettati per adattarsi a queste differenze, ad esempio,
regolando dinamicamente il numero di epoche o il batch size in base alle
risorse del dispositivo. Questo adattamento mira a garantire che anche i
Client con risorse limitate possano contribuire all’aggiornamento del
modello globale, mantenendo l’efficacia e l’efficienza dell’
addestramento distribuito su una rete eterogenea di dispositivi.

La durata di ogni round è dunque determinata dai Client più lenti
(strugglers). In un contesto dove il numero di Client è elevato e le
variazioni nella potenza computazionale e nel numero di iterazioni
locali dei dispositivi sono significative, questo approccio non è
ottimale ed impedisce al sistema di scalare. In una configurazione
Cross-Device, inoltre, i dispositivi sono per definizione inaffidabili e
potrebbero andare offline in qualsiasi momento.

Il problema principale è che FedAvg non permette ai partecipanti di
eseguire quantità variabili di lavoro locale in base ai vincoli dei loro
sistemi sottostanti (è comune semplicemente eliminare i dispositivi che
non riescono a calcolare le epoche *E* entro una finestra temporale
specificata) .

Nella configurazione classica di FedAvg, in cui non viene permesso ai
Client di eseguire una quantità variabile di lavoro e in cui non è
permessa l’aggregazione di risultati parziali, la convergenza del
modello è garantita . Queste assunzioni non sono però ragionevoli in un
contesto Cross-Device. Nel momento in cui vengono rilassate queste
assunzioni, il rischio è l’inconsistenza dell’obiettivo, cioè il modello
globale potrebbe convergere ad un punto stazionario di una funzione
obiettivo non corrispondente, che può essere arbitrariamente diversa
dall’obiettivo reale (come mostrato in
Figura <a href="#fig:objective_inconsistency" data-reference-type="ref"
data-reference="fig:objective_inconsistency">2.2</a>), oppure il modello
potrebbe non convergere affatto e causare aggiornamenti instabili.

<figure>
<img src="objective_inconsistency.png" id="fig:objective_inconsistency"
style="width:100.0%"
alt="Effetto di addestramenti locali con numero di epoche ed iperparamentri omogenei ed eterogenei. I quadrati verdi e i triangoli blu denotano i minimi degli obiettivi globali e locali, rispettivamente." />
<figcaption aria-hidden="true">Effetto di addestramenti locali con
numero di epoche ed iperparamentri omogenei ed eterogenei. I quadrati
verdi e i triangoli blu denotano i minimi degli obiettivi globali e
locali, rispettivamente.</figcaption>
</figure>

Nel FL la funzione obiettivo globale è definita come segue:
$F(x) = \\sum\_{i=0}^m{n_i \\cdot F_i(x)/n}$, dove *m* è il numero di
client e *n* = ∑<sup>*m*</sup>*n*<sub>*i*</sub> il numero totale di
campioni. Come dimostrato in la media standard dei modelli dopo
aggiornamenti locali eterogenei (di conseguenza FedAvg) porta alla
convergenza verso un punto stazionario, che non appartiene alla funzione
obiettivo originale *F*(*x*), ma ad una funzione obiettivo incoerente
*F*<sub>*e*</sub>(*x*), che può essere arbitrariamente diversa da
*F*(*x*) a seconda dei valori *τ*<sub>*i*</sub> (numero di epoche
locali). Per ottenere un’intuizione su questo fenomeno, si osservi la
Figura <a href="#fig:objective_inconsistency" data-reference-type="ref"
data-reference="fig:objective_inconsistency">2.2</a> che, se il Client 1
(*x*<sub>1</sub>) esegue più aggiornamenti locali rispetto al Client 2
(*x*<sub>2</sub>), allora l’aggiornamento *x*<sup>(*t*+1,0)</sup> si
allontana dal minimo globale vero *x*<sup>\*</sup>, dirigendosi verso il
minimo locale *x*<sub>1</sub><sup>\*</sup>

## Eterogeneità Statistica dei Dati

L’eterogeneità statistica dei dati determina in gran parte l’efficacia e
l’efficienza del processo di apprendimento federato. In un contesto
Cross-Device realistico i dati di addestramento sono distribuiti su un
ampio numero di dispositivi Client, ognuno dei quali raccoglie dati in
modo indipendente e con caratteristiche uniche, spesso non identicamente
distribuiti e non indipendenti.

Come anticipato nella precedente sezione, FedAvg ha dimostrato successo
empirico in configurazioni eterogenee purché vengano mantenute le sue
assunzioni stringenti. In caso in cui esse vengono a mancare metodi come
FedAvg hanno dimostrato di divergere in pratica quando i dati non sono
i.i.d. .

L’eterogeneità statistica influisce negativamente sulla convergenza
degli algoritmi di apprendimento federato e sulla qualità del modello
globale. Quando i dati dei Client sono non-i.i.d., le direzioni di
aggiornamento dei modelli locali possono essere molto diverse tra loro,
causando oscillazioni nella fase di aggregazione e rallentando la
convergenza . Inoltre, il modello globale potrebbe non rappresentare
adeguatamente i modelli locali ottimali, riducendo le prestazioni
complessive.

Un altro problema legato all’eterogeneità dei dati è la *fairness*,
poiché il modello appreso potrebbe mostrare un bias verso dispositivi
con una maggiore quantità di dati o, se si ponderano i dispositivi in
modo uguale, verso gruppi di dispositivi che compaiono più
frequentemente.

La considerazione sulla scelta tra un singolo modello globale e approcci
multi-modello nel Federated Learning è particolarmente rilevante,
soprattutto quando i dati locali sui dispositivi sono non-i.i.d. Se ogni
dispositivo è in grado di eseguire l’addestramento sui propri dati
locali, come richiesto dal Federated Learning, sorge naturalmente la
domanda: è sempre vantaggioso puntare a un unico modello globale? Un
singolo modello globale offre indubbiamente alcuni vantaggi. Ad esempio,
può essere distribuito ai Client privi di dati o con dati limitati,
garantendo una soluzione standardizzata e uniforme per tutti i
partecipanti. Questo approccio è inoltre utile quando le caratteristiche
dei dati dei dispositivi sono abbastanza simili, o quando la variabilità
statistica è minima, rendendo il modello generale e sufficientemente
efficace per tutti.

Tuttavia, nei casi in cui i dati sono fortemente non-i.i.d., un unico
modello globale può non essere ideale, poiché potrebbe non riuscire a
catturare le peculiarità dei dati locali di ciascun Client, con
conseguente degrado delle prestazioni complessive. In questi scenari,
approcci “multi-modello” come il Multi-Task Learning (MTL) e il
Meta-Learning possono essere più appropriati ed efficaci .

-   **Multi-Task Learning (MTL)**: Con MTL, ogni dispositivo addestra un
    modello che, pur condividendo parte della struttura con il modello
    globale, è adattato specificamente alle caratteristiche dei dati
    locali. In sostanza, MTL tratta ogni dataset locale come un task
    distinto, consentendo la creazione di modelli specifici per ciascun
    Client. Questo approccio permette a ciascun dispositivo di
    ottimizzare il modello per il proprio contesto, mantenendo una
    connessione con il modello globale ma permettendo una maggiore
    flessibilità e migliorando così le prestazioni quando i dati sono
    eterogenei.

-   **Meta-Learning**: Meta-Learning mira a creare un modello globale
    che non è semplicemente statico, ma è progettato per adattarsi
    rapidamente ai nuovi dati. In Federated Learning, un modello basato
    su Meta-Learning può apprendere una struttura globale e generica che
    ogni dispositivo può ulteriormente personalizzare sui propri dati,
    migliorando l’efficacia del modello per scenari con variabilità
    significativa.

Inoltre, la personalizzazione tramite tecniche come il fine-tuning
locale o il meta-learning permettono di adattare un modello globale ai
dati di ogni Client, garantendo una maggiore precisione a livello
locale.

Infine, l’eterogeneità statistica può avere effetti significativi anche
sulla privacy e sulla sicurezza nel Federated Learning. Quando i dati
locali dei dispositivi sono non-i.i.d. e altamente specifici,
l’aggiornamento dei modelli locali può riflettere particolari dettagli
distintivi dei dati di un Client. Questo può aumentare la vulnerabilità
a potenziali attacchi, come gli *attacchi di inversione del gradiente* .
In questi attacchi, un avversario potrebbe utilizzare le informazioni
contenute negli aggiornamenti del modello per risalire a dati sensibili
originari, sfruttando l’unicità dei pattern nei dati locali. Per
mitigare questi rischi, tecniche di protezione avanzate come la *privacy
differenziale* possono essere impiegate. La privacy differenziale
aggiunge rumore controllato agli aggiornamenti dei modelli, rendendo più
difficile per un attaccante inferire informazioni specifiche dai dati
originali di un Client senza compromettere troppo la qualità
dell’apprendimento globale. Questo approccio consente di proteggere le
informazioni sensibili, bilanciando privacy e accuratezza nel Federated
Learning .

## Architetture Software di Riferimento

Nel contesto del Federated Learning, l’architettura del sistema gioca un
ruolo cruciale nel determinare l’efficienza, la scalabilità e la
sicurezza del processo di addestramento distribuito. Esistono diverse
architetture proposte in letteratura per gestire i vari requisiti e per
affrontare le sfide del Federated Learning, ciascuna con i propri
vantaggi e svantaggi . Nel seguito descriviamo i quattro tipi principali
di architettura software.

  

##### Architettura centralizzata

L’architettura software prevalente nei sistemi di Federated Learning
attuali è quella centralizzata, in cui un singolo nodo Master coordina
l’intera rete di dispositivi periferici come mostrato in Figura 
<a href="#fig:arch-types" data-reference-type="ref"
data-reference="fig:arch-types">[fig:arch-types]</a>a. In questa
configurazione, il Master comunica con tutti i client, raccoglie i
modelli locali, esegue l’aggregazione e distribuisce il modello globale
aggiornato a tutti i dispositivi. Questa architettura è particolarmente
adatta a sistemi di piccole dimensioni, in quanto semplice da
implementare e gestire.

Tuttavia, l’architettura centralizzata presenta alcune limitazioni
significative:

-   *Singolo punto di fallimento*. Il Master è un elemento centrale che,
    per definizione, rappresenta un potenziale punto di vulnerabilità.
    Se il Master subisce un’interruzione, l’intero processo di
    apprendimento viene compromesso. Anche se organizzazioni o grandi
    aziende possono offrire server centrali affidabili per alcuni
    scenari, avere un Master sempre disponibile e potente potrebbe non
    essere fattibile o desiderabile in contesti più collaborativi,
    distribuiti o dinamici .

-   *Collo di bottiglia con molti Client*. Quando il numero di Client è
    elevato, il server Master potrebbe diventare un collo di bottiglia,
    riducendo l’efficienza e rallentando il processo di aggregazione.
    Lian et al. . hanno dimostrato che algoritmi decentralizzati possono
    superare i sistemi centralizzati in termini di prestazioni in
    scenari con numerosi client.

Va rilevato comunque che con un’attenta progettazione del sistema è
possibile mitigare l’impatto di un numero elevato di Client anche in un
sistema centralizzato. Ad esempio, tecniche di ottimizzazione della
comunicazione e gestione delle risorse possono migliorare la scalabilità
del sistema, riducendo la latenza e distribuendo il carico in modo più
efficiente .

##### Architettura gerarchica

L’architettura gerarchica di Federated Learning rappresenta
un’alternativa all’approccio centralizzato tradizionale, introducendo
nodi di coordinamento intermedi, spesso regionali, per gestire gruppi di
dispositivi edge,come mostrato in
Figura <a href="#fig:arch-types" data-reference-type="ref"
data-reference="fig:arch-types">[fig:arch-types]</a>b. In questa
configurazione, i dispositivi locali comunicano con i nodi regionali,
che a loro volta gestiscono l’aggregazione e l’aggiornamento dei modelli
per i dispositivi all’interno della propria area geografica o logica,
prima di inviare le informazioni al server centrale. Questo approccio
offre diversi vantaggi rispetto all’architettura centralizzata:

-   *Riduzione dei colli di bottiglia comunicativi*. I nodi regionali
    distribuiscono il carico di comunicazione, riducendo la congestione
    e i ritardi nelle trasmissioni di dati e aggiornamenti. Ciò rende
    l’architettura più scalabile, soprattutto per applicazioni di medie
    dimensioni, dove il numero di dispositivi è elevato ma non al
    livello delle grandi reti globali.

-   *Efficienza energetica e di banda*. Con l’elaborazione regionale, i
    dispositivi edge non devono sempre comunicare direttamente con il
    server centrale, diminuendo il consumo di banda e di energia,
    elementi cruciali nei contesti IoT e mobili.

Tuttavia, l’architettura gerarchica conserva ancora una parziale
centralizzazione, rappresentata dal server root della gerarchia, che
rimane un singolo punto di fallimento. Se il server centrale subisce
interruzioni, l’intero sistema potrebbe risentirne. Inoltre, questa
centralizzazione, anche se parziale, può sollevare preoccupazioni di
sicurezza e privacy, poiché la gestione delle informazioni aggregate è
ancora concentrata in un nodo centrale.

##### Architettura regionale

L’architettura regionale rappresenta un ulteriore passo verso la
decentralizzazione nel Federated Learning, eliminando completamente il
nodo di aggregazione centrale e affidando a nodi di aggregazione
regionali il compito di gestire l’aggregazione dei modelli e la
comunicazione tra i gruppi di dispositivi,
Figura <a href="#fig:arch-types" data-reference-type="ref"
data-reference="fig:arch-types">[fig:arch-types]</a>c. Questo design
presenta diversi vantaggi, tra cui:

-   *Miglioramento della robustezza*. L’assenza di un nodo centrale
    elimina il rischio di un singolo punto di guasto, aumentando la
    resilienza complessiva del sistema. In caso di interruzione di un
    nodo regionale, l’impatto è limitato al gruppo di dispositivi
    associati a quel nodo, senza compromettere l’intera rete.

-   *Addestramento del modello più localizzato*. In applicazioni dove le
    distribuzioni dei dati sono più simili tra i dispositivi vicini (ad
    esempio, in contesti geografici o di settore specifico), i nodi
    regionali possono aggregare modelli che rispecchiano meglio le
    caratteristiche locali. Questo porta a modelli più accurati per
    specifici gruppi di dispositivi, migliorando le performance senza
    dover adattare eccessivamente il modello globale.

Questa architettura si adatta particolarmente bene a scenari con reti
geograficamente distribuite o in cui le esigenze di privacy e sicurezza
richiedono una gestione più autonoma dei dati tra i gruppi.

##### Architettura decentralizzata

L’architettura decentralizzata nel Federated Learning sposta le
responsabilità di aggregazione direttamente sui dispositivi edge,
eliminando la necessità di nodi di coordinamento centrali o regionali,
come mostrato in
Figura <a href="#fig:arch-types" data-reference-type="ref"
data-reference="fig:arch-types">[fig:arch-types]</a>b. In questa
configurazione, ogni dispositivo contribuisce autonomamente
all’aggiornamento e alla condivisione dei modelli, distribuendo il
carico computazionale e di comunicazione su tutti i nodi partecipanti.
Questa struttura offre diversi vantaggi chiave, tra cui:

-   *Autonomia*. Poiché ogni dispositivo gestisce il proprio processo di
    addestramento e aggregazione, l’architettura è altamente autonoma. I
    dispositivi possono adattarsi rapidamente a cambiamenti locali,
    modificando il modello per rispondere a nuove condizioni ambientali
    o dati senza dipendere da un nodo centrale.

-   *Adattabilità*. Con l’aggregazione distribuita, l’architettura è
    altamente flessibile e adatta a sistemi dinamici e scalabili come le
    reti IoT, dove i dispositivi possono connettersi e disconnettersi
    frequentemente. La configurazione decentralizzata consente una
    maggiore resilienza e un ridotto impatto dei singoli guasti sui
    risultati complessivi del sistema.

Tuttavia, questa architettura richiede un’infrastruttura robusta e
presenta costi elevati. Ogni dispositivo deve avere la capacità di
eseguire l’addestramento locale e supportare la comunicazione per la
trasmissione dei modelli, caratteristiche che possono essere onerose in
termini di risorse computazionali, energetiche e di rete. Nonostante
questi requisiti, l’architettura decentralizzata rappresenta una
soluzione efficace per applicazioni su larga scala e scenari in cui la
privacy e la resilienza sono prioritarie, offrendo un sistema di
Federated Learning scalabile e meno vulnerabile ai colli di bottiglia e
ai punti di guasto singoli. Nell’apprendimento interamente
decentralizzato, l’interazione con un server centrale è completamente
sostituita da una comunicazione *peer-to-peer* tra i Client. In questa
architettura, la topologia di comunicazione è rappresentata da un grafo
connesso: i nodi del grafo sono i Client, mentre un arco tra due nodi
indica la presenza di un canale di comunicazione diretto tra due Client.
Il grafo della rete viene progettato per essere sparso e con un piccolo
grado massimo, limitando il numero di connessioni per ciascun nodo.
Questo design riduce il carico di comunicazione per ogni dispositivo,
poiché ogni Client invia e riceve messaggi solo da un numero ristretto
di peer vicini, riducendo l’onere computazionale e di rete. In questo
contesto, un round di apprendimento decentralizzato corrisponde a un
ciclo in cui ciascun client: a) esegue un aggiornamento locale del
modello usando i propri dati, e b) scambia il modello aggiornato o i
parametri con i client vicini nel grafo.

Una caratteristica fondamentale dell’apprendimento decentralizzato è
l’assenza di uno stato globale del modello, tipico delle architetture
federate classiche. Qui, non esiste un unico modello centrale verso cui
tutti i client convergono; al contrario, ogni client mantiene una
versione locale del modello, che evolve progressivamente grazie agli
scambi di parametri tra vicini. Questa decentralizzazione consente una
maggiore flessibilità e robustezza, soprattutto in contesti con elevata
dinamicità e dove i dispositivi si connettono e disconnettono
frequentemente.  

<div id="tab:architectures">

| **Caratteristica**              |  **C**   |    **G**     |  **R**   |    **D**     |
|:--------------------------------|:--------:|:------------:|:--------:|:------------:|
| Livello di centralizzazione     |   Alto   |   Moderato   |  Basso   |   Nessuno    |
| Scalabilità                     | Limitata |    Media     |   Alta   |  Molto alta  |
| Punto singolo di guasto         |    Sì    | Parzialmente | Ridotto  |      No      |
| Collo di bottiglia comunicativo |    Sì    | Parzialmente | Ridotto  |    Minimo    |
| Robustezza                      |  Bassa   |    Media     |   Alta   |  Molto alta  |
| Requisiti infrastrutturali      |  Bassi   |   Moderati   | Moderati |     Alti     |
| Complessità di gestione         |  Bassa   |    Media     |   Alta   |  Molto alta  |
| Dimensioni del sistema adatte   | Piccole  |    Medie     |  Grandi  | Molto grandi |

Confronto sintetico tra le diverse architetture di Federated Learning.
C: Architettura Centralizzata; G: Architettura Gerarchica; R:
Architettura Regionale; D: Architettura Decentralizzata.

</div>

Alla luce di quanto descritto per le architetture di riferimento nel
Federated Learning, la
Tabella <a href="#tab:architectures" data-reference-type="ref"
data-reference="tab:architectures">2.1</a> sintetizza le caratteristiche
peculiari di ognuna di esse.

## Apprendimento Federato Asincrono

L’approccio asincrono nel Federated Learning (AFL) rappresenta
un’innovazione importante per superare le difficoltà di sincronizzazione
e migliorare le prestazioni nei sistemi distribuiti. A differenza
dell’approccio sincrono, dove l’aggregatore deve attendere che tutti i
dispositivi completino l’addestramento locale prima di poter generare
una nuova versione del modello e avviare il round successivo, nell’AFL
il modello viene aggiornato ogni volta che un dispositivo invia i suoi
parametri aggiornati. Questo meccanismo offre alcuni vantaggi :

-   *Eliminazione degli strugglers*. Con l’AFL, i dispositivi più lenti
    (strugglers) non rallentano l’intero sistema, poiché il modello può
    essere aggiornato senza aspettare che tutti i client completino il
    loro addestramento. Questo migliora l’efficienza complessiva e
    consente un utilizzo ottimale delle risorse computazionali.

-   *Evoluzione continua del modello*. Il modello si aggiorna in modo
    incrementale man mano che riceve nuove informazioni, consentendo una
    rapida progressione dell’apprendimento e una maggiore adattabilità
    del modello ai dati in arrivo.

-   *Scalablità migliorata*. Ogni dispositivo può addestrare il modello
    secondo le proprie capacità computazionali e velocità, senza essere
    penalizzato dai tempi di elaborazione degli altri partecipanti.
    Questo rende AFL particolarmente adatto a sistemi su larga scala,
    dove i dispositivi hanno capacità eterogenee e partecipano in modo
    dinamico

<figure>
<img src="sync_async_arch.png" id="fig:sync-async" style="width:100.0%"
alt="Confronto schematico tra aggregazione sincrona ed asincrona" />
<figcaption aria-hidden="true">Confronto schematico tra aggregazione
sincrona ed asincrona</figcaption>
</figure>

Nonostante i vantaggi in termini di gestione dell’etoregeneità dei
dispositivi, l’approccio asincrono nel Federated Learning introduce
nuove sfide. In primo luogo, la gestione della *staleness*
(osolescenza), ovvero il rischio che gli aggiornamenti inviati dai
dispositivi possano risultare non allineati temporalmente con lo stato
attuale del modello globale. Questo può comportare un aumento
dell’errore durante la fase di aggregazione. Infatti, se un dispositivo
invia un aggiornamento basato su dati o su un modello molto obsoleto
rispetto all’ultima versione del modello aggregato, l’integrazione di
tale aggiornamento può causare aumenti dell’errore complessivo del
sistema. La mancata sincronizzazione potrebbe portare a situazioni in
cui il modello globale converge verso punti stazionari non ottimali ,
compromettendo la qualità del modello finale, o peggio ad un instabilità
degli aggiornamenti causando la divergenza del modello. Inoltre, poiché
i dispositivi non attendono gli altri per completare l’addestramento, si
rischia di accumulare modelli che non rappresentano accuratamente il set
di dati globale, specialmente in scenari di dati non-i.i.d.

La staleness rappresenta un’importante sfida legata alla tempistica
degli aggiornamenti dei modelli. Quando un dispositivo locale addestra
un modello sui propri dati, utilizza una versione del modello globale
che è stata inviata dal server centrale in un determinato momento.
Tuttavia, a causa di vari fattori come differenze nelle capacità di
calcolo, nella disponibilità della rete e nei tempi di completamento
dell’addestramento, ci possono essere dei ritardi nell’invio degli
aggiornamenti da parte dei diversi dispositivi.

Quando un dispositivo invia il proprio aggiornamento al server, questo
aggiornamento può essere basato su un modello che non è più l’ultimo
modello globale disponibile. Questo accade perché, nel tempo trascorso
dal momento in cui il dispositivo ha iniziato l’addestramento locale, il
server potrebbe aver ricevuto e aggregato aggiornamenti da altri
dispositivi, aggiornando quindi il modello globale. In altre parole,
l’aggiornamento che un dispositivo invia può riferirsi a uno stato del
modello che è stato superato da modifiche più recenti apportate da altri
dispositivi.

Una strategia comunemente utilizzata è quella di non aggregare
aggiornamenti inviati da client strugglers. Come esaminato in per
l’algoritmo FedAvg si nota che questa strategia, quando il numero di
strugglers è elevato, porta ad una rallentata convergenza ed una
possibile instabilità del modello.

L’effetto negativo degli strugglers è strettamente legato
all’etereogeneità statistica. Nel caso in cui i dati siano i.i.d. la
convergenza è più lenta ma comunque garantita. Come vedremo nella
sezione relativa alle sperimentazioni effettuate, la distribuzione delle
label dei dati sui vari Client non impatta in maniera significativa la
convergenza in presenza di strugglers.

Per mitigare le sfide introdotte dall’approccio asincrono esistono
tecniche ibride come FedBuff ed Hysync che vengono discusse nella
sezione successiva.

## Ottimizzazioni note

Per affrontare le problematiche del Federated Learning, sono stati
sviluppati diversi algoritmi e strategie di ottimizzazione che mirano a
migliorare la stabilità, l’efficienza e la robustezza del processo di
apprendimento federato.

##### FedProx - riduzione dell’eterogeneità statistica e di sistema

. FedProx è un’estensione dell’algoritmo FedAvg progettato
specificamente per affrontare l’eterogeneità dei sistemi e dei dati.
FedProx introduce un termine di regolarizzazione *μ* che penalizza la
differenza tra il modello locale di un Client e il modello globale.
Questo termine di prossimità aiuta a mantenere gli aggiornamenti locali
vicini al modello globale, anche quando i dati locali sono non-i.i.d.,
migliorando così la stabilità della convergenza. Inoltre, FedProx
permette ai client di eseguire un numero variabile di iterazioni locali
in base alle loro capacità computazionali, consentendo ai dispositivi
meno potenti di partecipare senza l’obbligo di completare un numero
fisso di epoche.

##### FedAvgM - miglioramento della convergenza con Momentum

FedAvgM . è un’estensione dell’algoritmo FedAvg che introduce il
concetto di Momentum lato Server per migliorare la stabilità e
l’efficienza della convergenza, in particolare nei casi di eterogeneità
statistica. Il Momentum è una tecnica comunemente utilizzata negli
algoritmi di ottimizzazione per accelerare la discesa del gradiente,
memorizzando una frazione della direzione di aggiornamento precedente e
sommando questo termine al gradiente corrente. Utilizzando il Momentum,
FedAvgM è in grado di attenuare le oscillazioni tra gli aggiornamenti
dei Client, portando a una convergenza più rapida e stabile. Questa
tecnica è particolarmente utile quando i dati sono non-i.i.d., poiché il
Momentum aiuta a smussare le differenze tra gli aggiornamenti locali e a
garantire una progressione più uniforme verso un modello globale
ottimale.

##### FedBuff - miglioramento delle aggregazioni asincrone con buffer

FedBuff è stato sviluppato per migliorare l’efficienza delle
aggregazioni asincrone, mitigando l’impatto degli aggiornamenti obsoleti
(staleness). FedBuff utilizza un buffer per accumulare e aggregare
aggiornamenti dai Client in modo più organizzato e strategico, così che
gli aggiornamenti vengano aggregati in momenti ottimali, riducendo al
minimo l’effetto negativo della staleness sugli aggiornamenti del
modello globale. Questo approccio migliora la stabilità del modello
anche quando i dispositivi partecipano in modo asincrono, ottimizzando
le risorse computazionali.

##### Hysync - gestione dell’eterogeneità di sistema con sincronizzazione ibrida

Hysync è una soluzione ibrida che combina gli approcci sincrono e
asincrono per affrontare l’eterogeneità di sistema e migliorare
l’efficienza delle aggregazioni. Hysync permette di eseguire la
sincronizzazione degli aggiornamenti in modo più flessibile, adattando
il processo di aggregazione alle capacità dei client. Questo approccio
riduce l’impatto dei client più lenti (strugglers) e garantisce che il
modello globale continui a evolvere senza significative attese,
mantenendo alta l’efficienza del processo di apprendimento.

##### FedNova - riduzione della variabilità con normalizzazione degli aggiornamenti

. FedNova è stato proposto per affrontare i problemi di convergenza
dovuti alla variabilità nelle iterazioni locali. FedNova introduce una
tecnica di normalizzazione degli aggiornamenti dei client per assicurare
che ogni contributo al modello globale sia equo e proporzionato al
lavoro svolto, indipendentemente dal numero di epoche effettuate. Questo
approccio è particolarmente utile per mitigare gli effetti negativi
dell’eterogeneità di sistema e statistica, garantendo che ogni
dispositivo contribuisca in modo equilibrato al modello globale,
migliorando così la convergenza.

##### SCAFFOLD - riduzione della varianza per l’eterogeneità statistica

. SCAFFOLD è un approccio progettato per ridurre la varianza introdotta
dall’eterogeneità statistica dei dati, introducendo una strategia di
correzione dei gradienti. SCAFFOLD utilizza una tecnica basata su una
stima della differenza tra il gradiente locale e quello globale per
stabilizzare gli aggiornamenti, riducendo significativamente la varianza
e migliorando la stabilità della convergenza. Questo approccio consente
una convergenza più rapida e precisa, rendendolo particolarmente
efficace nei contesti in cui i dati locali differiscono notevolmente tra
i dispositivi.

##### FedGroup - aggregazione per gruppi omogenei per una migliore personalizzazione

. FedGroup propone un approccio innovativo per migliorare la
personalizzazione e l’efficacia del Federated Learning in contesti
caratterizzati da alta eterogeneità dei dati e dei dispositivi. Invece
di applicare lo stesso modello globale a tutti i client, FedGroup
suddivide i dispositivi in gruppi omogenei basati su caratteristiche
comuni. Ogni gruppo elabora un modello intermedio più rappresentativo
per i propri dati, che viene poi raffinato con aggiornamenti locali.
Questo metodo riduce l’impatto della variabilità non-i.i.d. tra i
Client, promuovendo una personalizzazione più accurata e una migliore
convergenza del modello globale, preservando al contempo la privacy.

# Stato dell’Arte

Negli ultimi anni, il Machine Learning e l’Intelligenza Artificiale
hanno conosciuto un rapido sviluppo, portando a una crescente adozione
di queste tecnologie in numerosi ambiti applicativi. Parallelamente, la
necessità di preservare la privacy degli utenti e gestire efficacemente
dati altamente distribuiti ha focalizzato l’interesse verso il Federated
Learning (FL).

Per supportare questo approccio innovativo, sono stati sviluppati
numerosi framework e librerie, ciascuno progettato per affrontare
esigenze specifiche. Questi strumenti offrono funzionalità avanzate per
gestire scenari di Cross-Silo e Cross-Device, due categorie principali
che riflettono le diverse configurazioni del Federated Learning.

Come discusso nel
Capitolo <a href="#chap:background:architetture" data-reference-type="ref"
data-reference="chap:background:architetture">2.5</a>, Cross-Silo si
applica a scenari in cui i dispositivi partecipanti sono entità
organizzative o server controllati, con una disponibilità stabile e
risorse computazionali elevate tipiche in ambiti come la collaborazione
tra istituti o aziende; Cross-Device si adatta a scenari in cui una
vasta rete di dispositivi personali partecipa all’addestramento tipici
di ambienti altamente eterogenei, caratterizzati da connessioni
instabili, risorse limitate e partecipazione dinamica.

La selezione del framework più idoneo dipende strettamente dal tipo di
scenario e dalle specifiche esigenze applicative. Alcuni framework si
concentrano sull’ottimizzazione per scenari Cross-Silo, offrendo
strumenti per il coordinamento e la gestione centralizzata dei dati,
mentre altri sono progettati per affrontare la complessità e la
scalabilità richieste in ambienti Cross-Device, integrando meccanismi
per l’addestramento asincrono, la gestione dell’eterogeneità dei
dispositivi e la robustezza agli errori.

## Framework consolidati

Con lo sviluppo del Federated Learning, sono stati introdotti numerosi
framework sul mercato, tra cui TensorFlow Federated (TFF) , Flower ,
FederatedScope , PySyft , FedML  e OpenFL . Tuttavia, la maggior parte
di questi framework si concentra principalmente sulla qualità del
modello appreso, piuttosto che sulle ottimizzazioni legate alle
prestazioni dell’infrastruttura distribuita e delle comunicazioni
necessarie al processo federato. Molti di questi strumenti sono
progettati per scenari Cross-Silo, mentre solo alcuni, come
FederatedScope e Flower, sono in grado di affrontare le complessità di
sistemi Cross-Device, caratterizzati da dispositivi non affidabili e
operanti su larga scala.

Una proprietà in comune tra questi framework è il pattern architetturale
master slave. Seguendo le linee del primo algoritmo di FL FedAvg questi
framework si sono strutturati per seguire il suo approccio sincrono e
coordinato dove esiste un entità centrale/decentralizzata che controlla
e coordina i dispositivi federati. Un framework che cerca di discostarsi
da questa rigidità architetturale è FederatedScope il quale è
implementato con un’architettura basata su eventi, ampiamente utilizzata
nei sistemi distribuiti. Con questa architettura, un paradigma di FL può
essere definito come coppie di eventi ed azioni: i partecipanti
attendono determinati eventi (ad esempio, i parametri del modello
vengono trasmessi ai client) per attivare i gestori corrispondenti (ad
esempio, l’addestramento dei modelli basato sui dati locali). Pertanto,
gli utenti possono esprimere i comportamenti di server e client dalla
proprira prospettiva in modo indipendente, piuttosto che in modo
sequenziale da una prospettiva globale. Questa flessibilità permette di
creare un sistema in grado di supportare scenari con un grande
quantitativo di client e di definire diversi paradigmi di FL come
l’apprendimento asincrono in maniera triviale.

Nel seguito descriviamo brevemente i principali framework di federated
learning ad oggi disponibili.

##### TensorFlow Federated (TFF)

. Sviluppato da Google, è uno dei framework più consolidati per il FL,
costruito sull’ecosistema TensorFlow . Offre building-block per
l’implementazione di modelli, computazioni federate e gestione dei
dataset in scenari Cross-Silo grazie alla sua capacità di gestire dati
distribuiti in modo sicuro. TFF supporta la simulazione su più macchine
e la creazione di architetture distribuite, ma il suo utilizzo reale in
ambienti distribuiti su larga scala richiede ulteriori configurazioni e
non è ottimizzato di default per tali scenari. TFF presenta alcune
limitazioni. Ad esempio, non integra direttamente meccanismi di privacy
avanzata come la *Differential Privacy* (DF) o la *Secure Multi-Party
Computation* (SMPC), ma fornisce le basi per implementare tali tecniche.
Gli sviluppatori devono configurare manualmente le strategie di privacy
desiderate. Inoltre TFF non offre strumenti nativi per gestire attacchi
come la manomissione dei dati (data poisoning) o la manipolazione degli
aggiornamenti dei client (model poisoning). Pertanto, l’assenza di
meccanismi predefiniti per mitigare questi attacchi rende vera questa
parte dell’affermazione.

##### PySyft

. Sviluppato da OpenMined, introduce funzionalità avanzate per la
protezione della privacy, tra cui SMPC e DF. Questo framework,
compatibile sia con PyTorch che con TensorFlow , consente la
distribuzione su macchine singole o reti di nodi, utilizzando WebSocket
per la comunicazione. PySyft è particolarmente utile per sviluppare
applicazioni in cui la sicurezza dei dati è prioritaria.

##### SecureBoost

. E’ un framework specializzato nella costruzione di alberi di decisione
rinforzati in scenari con dataset partizionati verticalmente.
Utilizzando strategie di cifratura, SecureBoost  consente a diverse
parti di collaborare senza rivelare informazioni sensibili, garantendo
un’elevata accuratezza in contesti distribuiti.

##### FederatedScope e FedML

. Offrono ulteriori livelli di flessibilità. FederatedScope, grazie a
un’architettura basata su eventi, supporta strategie sincrone e
asincrone, simulazione di attacchi e protezione della privacy. FedML,
invece, integra protocolli di comunicazione come gRPC unito a Protocol
Buffers (brevemente gRPC/Proto), MPI e MQTT , adattandosi a diverse
configurazioni, da dispositivi IoT a infrastrutture Cross-Silo ad alte
prestazioni. Con i suoi moduli FedML-core e FedML-API, questo framework
permette la simulazione autonoma, il calcolo distribuito e
l’addestramento su dispositivo.

##### LEAF

. Si distingue come strumento di benchmarking per il FL, fornendo
dataset distribuiti e meccanismi di partizionamento utili per valutare
le performance di altri framework.

##### OpenFL

. OpenFL è un framework open-source progettato per il Federated Learning
(FL), che supporta sia il training distribuito che la protezione della
privacy dei dati. Si distingue per la sua architettura modulare, che
consente agli utenti di personalizzare facilmente i componenti del
sistema e di integrare diversi algoritmi di apprendimento federato.
OpenFL è compatibile con framework di deep learning popolari come
TensorFlow e PyTorch , e offre una gestione avanzata della comunicazione
tra nodi, riducendo la latenza e ottimizzando l’efficienza del sistema
distribuito. È particolarmente adatto per scenari in cui le performance
devono essere bilanciate con la protezione dei dati sensibili, come
nelle applicazioni in ambito sanitario o finanziario.

##### Flower

. Flower è un framework federato flessibile che supporta l’addestramento
distribuito su larga scala. Progettato per essere altamente modulare,
Flower consente agli utenti di implementare e testare facilmente diverse
strategie di apprendimento federato, come la sincronizzazione globale e
le tecniche di aggregazione personalizzate. Flower è compatibile con
framework di machine learning come TensorFlow , PyTorch e scikit-learn,
e offre una comunicazione efficiente tra client e server tramite gRPC .
Il suo design permette l’integrazione con diverse infrastrutture
hardware e software, rendendolo ideale per scenari con dispositivi edge,
IoT, o applicazioni su cloud. Flower è noto per la sua capacità di
bilanciare flessibilità, facilità d’uso e scalabilità.  
  
Tutti i framework FL riportati supportano una modalità di simulazione,
che consente di sperimentare e fare debugging di un sistema federato
localmente. Tuttavia, non è scontato che supportino anche una modalità
distribuita orientata al mondo reale, come ad esempio TFF, SecureBoost e
OpenFL. Da questa prospettiva, il framework FL più limitato è LEAF:
questo software è progettato esplicitamente per essere utilizzato solo
per scopi di benchmarking.

<div id="tab:frameworks">

| **Framework**  | **Cross-**  | **Scenario** |     **Protocol**      | **Implem.** |
|:--------------:|:-----------:|:------------:|:---------------------:|:-----------:|
|      TFF       |    silo     |  sim., real  |      gRPC/proto       |   Python    |
|     PySyft     | silo/device |  sim., real  |      Websockets       |   Python    |
|  SecureBoost   |    silo     | simu., real  |      gRPC/proto       |   Python    |
| FederatedScope | silo/device |  sim., real  |      gRPC/proto       |   Python    |
|      LEAF      |    silo     |     sim.     |          \-           |   Python    |
|     FedML      |    silo     |  sim., real  | gRPC/proto, MPI, MQTT | Python/C++  |
|     OpenFL     |    silo     |  sim., real  |      gRPC/proto       |   Python    |
|     Flower     | silo/device |  sim., real  |      gRPC/proto       |   Python    |

tabella riepilogativa di confronto tra i framework discussi

</div>

#### Comunicazione

Un elemento centrale nel design di framework di Federated Learning è
come viene implementata la comunicazione tra le differenti entità. La
maggior parte degli strumenti, tra cui TFF, FedML, Flower e
FederatedScope, utilizza protocolli come gRPC/Proto per la comunicazione
tra Client e Server. Questo protocollo garantisce buone prestazioni,
riducendo la latenza e gestendo in modo efficiente le richieste e
risposte in scenari federati.

Framework come PySyft si differenziano adottando alternative come
WebSocket, che fornisce connessioni bidirezionali persistenti. Questo
approccio è particolarmente adatto per ottimizzare la comunicazione in
tempo reale in scenari con requisiti di latenza ridotti.

FedML, invece, estende il supporto a protocolli aggiuntivi come MPI e
MQTT, garantendo una maggiore adattabilità:

-   **MPI** (Message Passing Interface) è ideale per ambienti ad alte
    prestazioni come i contesti Cross-Silo, in cui le reti e le risorse
    computazionali sono robuste (ad esempio, cluster con reti ad alte
    prestazioni).

-   **MQTT** (Message Queuing Telemetry Transport) si rivolge ai
    dispositivi IoT, offrendo comunicazioni a basso consumo energetico e
    a larghezza di banda ridotta, rendendolo adatto a scenari
    Cross-Device.

La scelta del framework e del protocollo di comunicazione dipende
fortemente dalle caratteristiche dello scenario applicativo e dai
vincoli di privacy, scalabilità e risorse disponibili che si vogliono
ottenere. Nella Tabella
<a href="#tab:frameworks" data-reference-type="ref"
data-reference="tab:frameworks">3.1</a> sono riportate le
caratteristiche principali di ciascun framework di Federated Learning,
con l’obiettivo di evidenziarne le differenze e fornire un confronto
diretto delle loro funzionalità, come il supporto per Cross-Silo o
Cross-Device, lo scenario applicativo, i protocolli utilizzati ed il
linguaggio di programmazione utilizzato per l’implementazione.

#### Apprendimento asincrono

L’apprendimento federato asincrono rappresenta un’evoluzione
significativa rispetto all’approccio sincrono tradizionale, offrendo
vantaggi in termini di efficienza, scalabilità e gestione
dell’eterogeneità tra i dispositivi partecipanti. In un ambiente
asincrono, i dispositivi non devono attendere il completamento degli
aggiornamenti di tutti i client prima di procedere, riducendo i ritardi
e migliorando l’utilizzo delle risorse computazionali. Tuttavia,
nonostante questi vantaggi, il supporto per l’apprendimento asincrono
nei framework di Federated Learning è ancora piuttosto limitato.

La maggior parte dei framework attualmente disponibili, come TensorFlow
Federated, FedML e OpenFL, sono stati progettati per operare in modalità
sincrona e coordinata, con un forte focus sulla gestione dei round di
comunicazione centralizzati. Questa struttura impone una
sincronizzazione tra i dispositivi, che risulta incompatibile con i
requisiti dell’apprendimento asincrono.

Per supportare algoritmi asincroni, è necessaria una revisione
significativa delle architetture sottostanti, in particolare dei
meccanismi di aggregazione e comunicazione. Anche framework flessibili
come Flower, sebbene offrano alcune funzionalità per scenari asincroni,
richiedono modifiche sostanziali per un pieno supporto di questa
modalità.

Attualmente, pochi framework supportano esplicitamente l’apprendimento
asincrono:

-   FedML: Include moduli sperimentali per il supporto di scenari
    asincroni, ma l’integrazione è limitata a contesti specifici e non è
    ottimizzata per grandi reti distribuite.

-   Flower: Consente una configurazione flessibile delle comunicazioni,
    il che facilita una certa implementazione di algoritmi asincroni.
    Tuttavia, manca un’infrastruttura nativa per la gestione di
    aggiornamenti asincroni complessi.

La ricerca in questo campo sta esplorando nuove architetture e
protocolli di comunicazione per migliorare il supporto all’asincronismo,
affrontando sfide come la *staleness* degli aggiornamenti,
l’aggregazione dinamica e la consistenza del modello globale. Con lo
sviluppo di questi approcci, si prevede un’espansione del supporto per
l’apprendimento asincrono nei framework futuri.

## Considerazioni

Il panorama attuale del Federated Learning offre una vasta gamma di
strumenti progettati per soddisfare esigenze applicative diverse.
Framework come TensorFlow Federated, PySyft, FATE, Flower, OpenFL e
FedML si distinguono per i loro differenti livelli di astrazione e
facilità d’uso. Alcuni sono ottimizzati per implementazioni rapide e
pronte all’uso, mentre altri si concentrano sulla personalizzazione e
flessibilità, rendendoli adatti a utenti con competenze avanzate.
Tuttavia, la gestione dell’asincronia nei contesti Cross-Device,
caratterizzati da dispositivi eterogenei e connessioni instabili, rimane
una sfida tecnica significativa.

In questo lavoro di tesi abbiamo voluto semplificare l’accesso a queste
tecniche, offrendo soluzioni che possano essere facilmente implementate
anche da un pubblico più ampio. Questo approccio non solo facilita la
creazione di sistemi distribuiti ed eterogenei, ma promuove anche
l’adozione del FL in scenari dove la privacy e la scalabilità sono
requisiti importanti. In questo modo, si contribuisce a rendere il
Federated Learning più accessibile riducendo le barriere tecniche per il
suo utilizzo.

# Architettura software del framework proposto

Questo capitolo presenta l’architettura software del framework
sviluppato in questa tesi, concepito per abilitare scenari di Federated
Learning (FL) in ambienti altamente eterogenei e distribuiti. Il
framework si distingue per il suo approccio modulare e il livello di
astrazione che rende il suo utilizzo estremamente intuitivo per l’utente
finale. L’obiettivo principale è quello di offrire un sistema che, pur
gestendo la complessità intrinseca dell’apprendimento federato, mantenga
un’interfaccia d’uso semplice e familiare, paragonabile a quella dei
classici scenari di apprendimento centralizzato ed in alcuni casi ancora
più intuitiva.

#### Scopi e Obiettivi

Il framework proposto si prefigge di:

-   Semplificare l’adozione del Federated Learning: consentendo agli
    utenti di concentrarsi solo sugli aspetti fondamentali, come il
    flusso di dati di training, il modello da allenare ed una
    configurazione minimale.

-   Gestire ambienti eterogenei: adattandosi a dispositivi con risorse
    computazionali e di rete variabili, sfruttando un approccio
    "resource-driven".

-   Massimizzare la modularità e la flessibilità: offrendo
    un’architettura facilmente estendibile per supportare futuri
    miglioramenti, come approcci decentralizzati o algoritmi
    personalizzati.

-   Ottimizzare le prestazioni: il framework è scritto in C, un
    linguaggio ad alte prestazioni, e adotta pattern di
    parallelizzazione e I/O non bloccante/asincrono, garantendo
    efficienza nella gestione delle comunicazioni e nell’elaborazione
    dei modelli, anche in scenari con numerosi client e grandi volumi di
    dati.

-   Migliorare l’efficienza della comunicazione: attraverso l’uso del
    protocollo TCP ottimizzato, messaggi in formato personalizzato e
    tecniche avanzate come quantizzazione e compressione dei modelli, il
    framework riduce significativamente la latenza e il consumo di
    banda. Questi meccanismi permettono di supportare trasferimenti
    rapidi e scalabili anche in presenza di grandi quantità di client e
    aggiornamenti frequenti.

<figure>
<img src="assets/astrazione1.png" id="fig:archAstr" style="width:100.0%"
alt="Schema astratto dal punto di vista dello sviluppatore" />
<figcaption aria-hidden="true">Schema astratto dal punto di vista dello
sviluppatore</figcaption>
</figure>

Grazie al suo design, il framework consente agli sviluppatori di
sfruttare le potenzialità del Federated Learning senza la necessità di
addentrarsi nella complessità tecnica dell’implementazione sottostante.
Questa filosofia "plug-and-play" è resa possibile da un alto livello di
astrazione. L’idea originale è quella di fornire una soluzione "out of
the box" in cui l’utente fornisce solamente 3 input
(Figura <a href="#fig:archAstr" data-reference-type="ref"
data-reference="fig:archAstr">4.1</a>): Il flusso di dati per il
training locale; Il modello da allenare; Ed una configurazione
contenente informazioni di connessione, la strategia di aggregazione da
utilizzare. Per non limitare la personalizzazione dei processi di
apprendimento e quindi la flessibilità stessa di questo prodotto in
realtà l’utente può fornire più configurazioni per esempio ridefinendo
alcune funzioni logiche chiave del processo di apprendimento.

## Resource driven learning

Il framework proposto da questa tesi si discosta molto dall’approccio
utilizzato da altre soluzioni disponibili discusse nella sezione stato
dell’arte (cf. Capitolo <a href="#chap:sota" data-reference-type="ref"
data-reference="chap:sota">3</a>). Nel Federated Learning tradizionale è
il server centrale che seleziona un sottoinsieme di dispositivi
disponibili ed avvia e coordina il processo di apprendimento. Il
processo federato viene organizzato in round, un round termina quando
tutti i client (o una certa soglia, a seconda dell’implementazione)
hanno inviato il proprio modello locale. Al termine di un round viene
calcolato il nuovo modello globale e la procedura ricomincia. La nostra
proposta sposta il controllo della partecipazione e dell’invio dei
modelli locali ad i client stessi che li producono. Riteniamo che sia il
client stesso l’entità che possa meglio valutare o meno quando sia il
momento migliore per partecipare al miglioramento del modello globale.
Per esempio quando si ritiene di aver raccolto una quantità di dati
rilevante, oppure quando le risorse computazionali lo permettono (per
esempio il livello di carica della batteria per un dispositivo mobile o
IoT). Sono dunque le risorse dei dispositivi federati che guidano il
processo di allenamento e non un’entità centrale: non è più
l’aggregatore centrale che seleziona i client per partecipare
all’allenamento del modello globale, ma i client che contribuiscono
quando possono al suo miglioramento. Per permettere questo approccio
"Resource driven" il sistema deve essere necessariamente asincrono, il
che, come discusso in precedenza nella sezione background
\[<a href="#fig:sync-async" data-reference-type="ref"
data-reference="fig:sync-async">2.3</a>\], porta con sé vantaggi di
scalabilità ma anche significativi problemi di instabilità in caso di
eterogeneità statistica. Nella figura
 <a href="#fig:flow-client" data-reference-type="ref"
data-reference="fig:flow-client">[fig:flow-client]</a> viene mostrato il
processo di apprendimento dal punto di vista di un client federato, come
si vede il processo di allenamento scaturisce da un evento locale, che
sia temporale e non. L’evento "Force Sync" è un evento inviato dal
server (questo può essere disabilitato in fase di configurazione) prima
dell’aggregazione, come si vede nella figura
 <a href="#fig:flow-server" data-reference-type="ref"
data-reference="fig:flow-server">[fig:flow-server]</a>, e server e serve
per forzare l’arresto forzato della fase di training. Questo evento
viene utilizzato per l’implementazione di algoritmi come HySync
\[<a href="#fedopt:hysync" data-reference-type="ref"
data-reference="fedopt:hysync">2.7.0.0.4</a>\]

<span id="fig:flow-client" label="fig:flow-client"></span>
<img src="assets/flow-client.png" style="width:15cm" alt="image" />

<span id="fig:flow-server" label="fig:flow-server"></span>
<img src="assets/flow-server.png" style="width:9cm" alt="image" />

Il processo che invece segue il server è totalmente passivo e guidato
dalla ricezione di aggiornamenti da parte dei client. Anche se fuori
dallo scopo di questo framework, un classico approccio sincrono, come
quello proposto dalla maggior parte delle soluzioni esistenti, potrebbe
essere simulato. Per prima cosa il server deve essere configurato in
modo che aggreghi i modelli quando un numero minimo di aggiornamenti è
stato ricevuto. Questa è una configurazione nativamente supportata in
quanto il server utilizza una strategia di bufferizzazione degli
aggiornamenti. I client, come il server, non subiscono grandi
cambiamenti: è sufficiente sostituire l’evento che da inizio
all’allenamento del modello con la notifica della presenza di un nuovo
modello globale inviata dal server. Qui sotto viene riportato uno
pseudo-codice che mostra quanto appena detto.

``` python
loop:
    wait_for_new_model()
    model = get_latest_global_model()
    local_model = train(model, data)
    send_update(local_model)
```

## Architettura Software Server

L’obiettivo di questa tesi è progettare un framework capace di gestire
scenari di Federated Learning in ambienti altamente eterogenei e
distribuiti, che possano coinvolgere un elevato numero di client.
Sebbene il nostro approccio iniziale impieghi un’architettura
centralizzata, essa è progettata in modo modulare per supportare future
implementazioni scalabili, affidabili e resilienti. L’architettura
centralizzata viene adottata in questa fase come un punto di partenza
per testare la fattibilità della soluzione proposta. In futuro, il
framework potrà evolversi verso un’architettura federata più
decentralizzata, con nodi più distribuiti e con sistemi di high
availability. Vedi sezione sulle
architetture federate <a href="#chap:background:architetture" data-reference-type="ref"
data-reference="chap:background:architetture">2.5</a>.

##### Linguaggio e Design

La scelta del linguaggio di implementazione del server è cruciale per
ottenere prestazioni elevate, specialmente in un contesto di Federated
Learning dove la gestione delle comunicazioni e l’elaborazione dei
modelli richiedono grande efficienza. A differenza di molte soluzioni
esistenti che utilizzano Python (cf. Capitolo
 <a href="#chap:sota" data-reference-type="ref"
data-reference="chap:sota">3</a>), il nostro framework è implementato in
C, un linguaggio di sistema noto per le sue performance. L’uso di C ci
permette di ridurre al minimo il tempo di esecuzione per le operazioni
più critiche e consente una gestione più fine della memoria e delle
risorse. L’architettura del software è progettata con un focus
sull’adozione di pattern orientati alle performance, come il pattern
Producer-Consumer, per gestire le code di aggiornamenti dei modelli e
migliorare l’efficienza nelle operazioni di I/O e pattern Data Parallel
per ridurre latenze e migliorare il throughput. Nella
Figura <a href="#fig:server-structure" data-reference-type="ref"
data-reference="fig:server-structure">4.2</a> è illustrata ad alto
livello la struttura del sistema software lato server, le cui componenti
vengono introdotte nei paragrafi successivi.

<figure>
<img src="assets/server-structure.png" id="fig:server-structure"
style="width:100.0%" alt="Architettura software del Server." />
<figcaption aria-hidden="true">Architettura software del
Server.</figcaption>
</figure>

#### Comunicazione Client-Server

Per la gestione della comunicazione tra Server e Client, il sistema
adotta il protocollo TCP, che garantisce una trasmissione affidabile dei
messaggi. La scelta di TCP è motivata dalla necessità di avere un canale
sicuro e stabile per l’invio di modelli e aggiornamenti tra Client e
Server. Ogni messaggio di comunicazione tra il Server e i Client è
codificato secondo un formato definito ad-hoc, per ottimizzare
l’efficienza e ridurre la latenza. Per ottimizzare la gestione di un
numero elevato di client e richieste contemporanee, abbiamo adottato
un’architettura basata su eventi e I/O non bloccante, sfruttando l’API
*epoll* , una funzione della libreria standard di Linux. Epoll è
particolarmente vantaggiosa in scenari ad alta concorrenza, in quanto
consente di monitorare un numero elevato di file descriptor senza
necessità di operazioni di polling ripetute, migliorando
significativamente le performance rispetto ad altri metodi come `select`
e `poll`. L’uso di epoll permette di gestire in modo efficiente le
connessioni I/O, notificando il Server solo quando ci sono dati da
leggere o scrivere, evitando il blocco del processo principale e
migliorando la reattività del sistema. Inoltre, epoll è altamente
scalabile, in quanto supporta la gestione di migliaia di connessioni
simultanee con un overhead minimo. Per incrementare ulteriormente la
capacità del server di gestire un numero maggiore di Client, è
possibile, come illustrato in
Figura <a href="#fig:server-structure" data-reference-type="ref"
data-reference="fig:server-structure">4.2</a>, utilizzare più thread
(tipicamente denominati *Workers*) per distribuire il carico di lavoro.
In questo contesto, il kernel Linux consente di accettare connessioni
sulla stessa porta da più thread senza necessità di meccanismi di
sincronizzazione complessi, grazie al parametro `SO_REUSEADDR`. Questo
parametro permette a diversi thread di legarsi alla stessa porta,
rendendo possibile l’uso di un pool di thread per gestire in parallelo
le connessioni in ingresso. Le connessioni vengono distribuite dal
sistema operativo in modo Round Robin, assegnando a ciascuna connessione
un thread specifico per l’elaborazione. In questo modo, ogni connessione
è completamente isolata e gestita da un singolo thread, garantendo una
gestione parallela e scalabile delle richieste. Questa architettura
rende il sistema estremamente efficiente su architetture multi-core ed è
capace di scalare orizzontalmente, consentendo di gestire centinaia di
migliaia di connessioni simultanee in modo fluido e reattivo.

#### Aggregazione Asincrona dei Modelli

L’aggregazione dei modelli avviene in modo asincrono. Quando arrivano
aggiornamenti dai Client, questi vengono dapprima decompressi e
normalizzati, per poi essere immagazzinati in un buffer. La strategia di
buffering decide quando è il momento opportuno per procedere con
l’aggregazione dei modelli, in base alla quantità di aggiornamenti
ricevuti e alla logica di gestione del flusso. Una volta che il momento
giusto per l’aggregazione è stato identificato, vengono assegnati i pesi
ai modelli in base all’algoritmo di ottimizzazione federata utilizzato.
Ad esempio in FedAvg il peso di ciascun modello è proporzionale al
numero di campioni su cui è stato addestrato. I pesi possono dipendere
da altri fattori legati alla specifica strategia di ottimizzazione. Dopo
l’assegnazione dei pesi, si esegue una media ponderata dei modelli.

Sia la fase di aggregazione che quelle di pre-processing e averaging dei
modelli possono essere eseguite in parallelo. La fase di pre-processing
può adottare un pattern che combina i costrutti *Farm* e *Pipelining*,
mentre la fase di averaging può essere gestita tramite il pattern
*Reduce*. Questi approcci consentono di ottimizzare l’elaborazione dei
modelli, sfruttando al massimo la parallellizzazione per migliorare le
performance complessive.

Per gestire grandi volumi di aggiornamenti e modelli di dimensioni
elevate, tutte le operazioni relative al modello fanno uso della
paginazione su disco utilizzando "Memory Mapped IO". Questo approccio
consente al sistema di evitare il sovraccarico della memoria principale,
permettendo una gestione più efficiente delle risorse, soprattutto
quando si trattano dati di grandi dimensioni. Quando vengono aggregati i
modelli per esempio questi sono salvati sul disco ed una pagina alla
volta vengono aggregati e viene generato il nuovo modello, questo
permette al server di poter gestire grandi quantità di aggiornamenti
senza mai eccedere la quantità di memoria disponibile.

#### I/O Overhead e Gestione della Memoria

La comunicazione tra i sottosistemi di comunicazione e aggregazione
avviene tramite strutture dati in memoria condivisa. Ogni aggiornamento
viene ricevuto suddiviso in chunk ordinati per ridurre la latenza nella
gestione dei pacchetti e prevenire il sovraccarico della memoria. Ogni
chunk dopo essere ricevuto viene quindi inserito in una coda di
elaborazione.

Un thread specializzato gestisce il trasferimento degli aggiornamenti
sul disco, utilizzando I/O asincrono per ridurre l’overhead derivante
dal context switch tra il kernel e lo spazio utente, nonché dall’uso
dell’I/O bloccante. Una volta che un aggiornamento è completamente
scritto su disco (tutti i suoi chunk sono stati salvati), il suo
identificatore viene inserito nella coda degli aggiornamenti, che viene
successivamente letta dal sottosistema di aggregazione per essere
processata.

Un approccio simile viene adottato per l’invio dei modelli ai client, al
fine di ridurre l’overhead associato alla compressione e invio dei
modelli. Quando una richiesta di modello viene ricevuta, essa viene
inserita nella coda del gestore dei modelli, mentre il socket viene
momentaneamente disabilitato. Il gestore dei modelli possiede una cache,
che contiene modelli precedentemente salvati nel file system. Se il
modello richiesto è già presente nella cache, viene immediatamente
inviato al client. Se il modello non è disponibile, la richiesta viene
aggiunta alla coda di compressione. Una volta completata la
compressione, la richiesta di invio del modello viene riaggiunta alla
coda del gestore dei modelli per l’invio effettivo.

La cache gioca un ruolo cruciale nelle performance del sistema. Ad
esempio, quando il modello globale viene aggiornato, è molto probabile
che si ricevano richieste per lo stesso modello da più client, e l’uso
della cache permette di ridurre significativamente i tempi di risposta e
l’overhead computazionale. In prospettiva futura, questo sottosistema
potrebbe essere esternalizzato dal Server centrale e implementato
attraverso una Content Delivery Network (CDN), migliorando ulteriormente
la distribuzione e la scalabilità del sistema.

#### Eventi di sincronizzazione

La gestione dell’invio eventi come ForceSync e NewGlobalModel viene
gestitia attraverso un pattern publish-subscribe. Esiste un messaggio
speciale di upgrade che rimuove la connessione dalla normale gestione di
epoll e aggiunge il socket ad una lista di listeners. Cosí facendo viene
ceduta l’ownership della connessione al thread di aggregazione.

#### Configurazione e Personalizzazione

La configurazione e la personalizzazione del framework per adattarlo a
qualsiasi scenario ad algoritmo di federated learning avviene tramite la
definizione di due funzioni da parte dell’utilizzatore.

``` objectivec
typedef struct
{
    uint8_t type; // 0: NO, 1: YES, 2: WAIT
    time_t until;
} agg_config_t;

agg_config_t should_aggregate_models(updates_t updates, time_t *last_aggregation, func_t force_sync)

double get_update_weight(model_info_t local_model, model_t gobal_model)
```

La funzione should_aggregate_models definisce la strategia di
bufferizzazione. L’utente in base agli aggiornamenti pronti per essere
aggregati decide se aggregare subito i modelli, se non aggregare ed
aspettare aggiornamenti successivi o se aspettare un lasso di tempo ed
eseguire l’aggregazione se nessun aggiornamento viene ricevuto nel tempo
specificato. Una funzione force_sync viene passata per permettere
all’utente di mandare una notifica ad i client che non hanno ancora
terminato l’addestramento del modello. La funzione get_update_weight
invece definisce la strategia di aggregazione, riceve in input le
informazioni del modello locale ed il modello globale e ritorna il peso
che il modello locale deve avere nel passaggio successivo di averaging.
Il modello con se, oltre che ai pesi, contiene anche dei metadati che
possono essere utilizzati per personalizzare maggiormente il processo di
aggregazione. Per implementare un algoritmo di FedAvg per esempio è
necessario un metadato che contenga la dimensione del dataset su cui il
modello locale è stato allenato. Per garantire la correttezza dei
metadati è necessario definire una funzione (is_valid_metadata) che può
essere utilizzata per validare il contenuto. Essendo questa una tesi che
mira a studiare la fattibilità della soluzione proposta per il momento
queste funzioni devono essere definite all’interno del codice sorgente,
ciò richiede la conoscenza del linguaggio C e la ricompilazione manuale
del progetto. Per il futuro l’idea però è quella di distribuire
l’applicazione pre compilata e di permettere la personalizzazione
attraverso linguaggi di scripting più ad alto livello facendo
affidamento a librerie come Extism che permettono all’utente di definire
queste funzioni in qualsiasi linguaggio l’utente preferisca (Python, JS,
C, C++, Java, Ruby, Go ... etc). Oltre che alla possibilità di
scripting, l’idea è che il framework supporti già nativamente le
strategie più comuni.

## Protocollo di comunicazione

Il protocollo di comunicazione client-server necessario per
l’implementazione della soluzione proposta in questo elaborato è molto
semplice e necessita di un numero limitato di messaggi. Per questa
ragione abbiamo optato per un’implementazione semplice che non faccia
utilizzo di tecnologie come gRPC, HTTP, Websocket, MPI etc. Per la
comunicazione tra client e server abbiamo dunque utilizzato dei socket
TCP con messaggi in formato non standard e ottimizzato per questo caso
d’uso. Nella
figura <a href="#fig:proto-message" data-reference-type="ref"
data-reference="fig:proto-message">4.3</a> viene mostrata la struttura
di un messaggio generico.

<figure>
<img src="assets/proto-message" id="fig:proto-message"
style="width:14cm"
alt="Struttura di un messaggio del protocollo di comunicazione" />
<figcaption aria-hidden="true">Struttura di un messaggio del protocollo
di comunicazione</figcaption>
</figure>

Il protocollo è costituito da 4 messaggi principali: Autenticazione,
Richiesta del modello globale, Invio di un aggiornamento, Sottoscrizione
al canale di notifica.

##### Formato dei modelli

Il framework proposto è progettato per essere completamente trasparente
per l’utente, consentendogli di utilizzare liberamente qualsiasi backend
di machine learning desideri, come PyTorch, TensorFlow o altri.
Tuttavia, ciascuno dei principali framework di machine learning utilizza
un formato proprietario per salvare lo stato dei modelli allenati, e
questi formati non sono ottimizzati per contesti di apprendimento
federato, in cui i pesi dei modelli vengono frequentemente trasmessi tra
dispositivi e dove l’uso efficiente della banda di rete è cruciale. In
un contesto di apprendimento federato, i pesi del modello devono essere
condivisi frequentemente tra i nodi partecipanti e il server centrale.
La continua trasmissione di modelli allenati rappresenta un carico
significativo per la larghezza di banda della rete, il che rende
necessaria una strategia che minimizzi il volume dei dati trasmessi.
Inoltre, per supportare algoritmi di ottimizzazione federata avanzati,
come Federated Averaging (FedAvg), è fondamentale arricchire il modello
con metadati che descrivano il contesto di addestramento, ad esempio la
dimensione del dataset utilizzato in ciascun nodo. Queste informazioni
consentono di attribuire un peso appropriato agli aggiornamenti inviati
da ciascun partecipante, migliorando così la precisione e l’efficacia
dell’ottimizzazione.

Per soddisfare questi requisiti, abbiamo scelto di sviluppare un formato
file personalizzato, ottimizzato sia per l’efficienza nell’uso della
banda di rete sia per l’integrazione di metadati e informazioni
strutturali, necessarie per una gestione efficace degli aggiornamenti
dei modelli nel contesto dell’apprendimento federato. In
Figura <a href="#fig:model-format" data-reference-type="ref"
data-reference="fig:model-format">4.4</a> è illustarta la struttura del
file.

<figure>
<img src="assets/model-format.png" id="fig:model-format"
style="width:9cm"
alt="Struttura del file che rappresenta lo stato di una rete ML" />
<figcaption aria-hidden="true">Struttura del file che rappresenta lo
stato di una rete ML</figcaption>
</figure>

Il formato del file è progettato con una struttura a due componenti
principali: un’intestazione e un corpo. Questa organizzazione permette
un accesso rapido alle informazioni chiave sul modello fin dal primo
byte trasmesso, consentendo un’elaborazione efficiente anche durante il
trasferimento (Questo permette di gestire gli aggiornamenti come una
stream di dati). L’intestazione contiene tutte le informazioni
necessarie per descrivere il modello, ed è strutturata in modo da non
essere compressa. Questa scelta permette al server di accedere a tutte
le informazioni essenziali non appena viene ricevuto il primo chunk di
dati, consentendo una gestione ottimizzata della rete e
un’interpretazione immediata del file. L’intestazione è composta da:

-   Informazioni sul file: versione del formato, dimensioni dei vari
    segmenti del file e flag che descrivono proprietà come la
    compressione, la quantizzazione e la presenza di sezioni opzionali.

-   Metadati: una sezione dedicata a informazioni di vario tipo, come
    stringhe, interi o float, che descrivono il contesto di
    addestramento del modello. Questi metadati possono includere
    dettagli come il numero di campioni usati nel training locale o la
    configurazione specifica del nodo. Questi dati consentono di
    calibrare correttamente gli aggiornamenti dei modelli,
    particolarmente utile per algoritmi di aggregazione come FedAvg, che
    richiedono informazioni sulla dimensione del dataset.

-   Descrizione della struttura dei tensori: una sezione opzionale che
    fornisce dettagli sulla struttura dei tensori, come dimensioni e
    layout. Questa sezione può essere disabilitata tramite un flag se il
    destinatario conosce già tali informazioni, riducendo così
    ulteriormente la quantità di dati trasmessi. Le descrizioni
    strutturali sono utili per garantire che le operazioni di
    aggregazione o fusione sui modelli siano effettuate con precisione,
    senza rischi di interpretazioni errate dei dati.

Il corpo del file contiene i valori dei tensori, ossia i pesi e i
parametri del modello, in un formato sequenziale organizzato in ordine
row-major. Questa disposizione permette di trattare i pesi come un unico
grande vettore, facilitando le operazioni di aggregazione come la media
tra i modelli: i parametri possono essere sommati direttamente, senza
dover conoscere dettagli specifici sulla struttura interna dei tensori.
Questa semplificazione rende il formato altamente flessibile e adatto ad
ambienti federati, dove le strutture dei modelli tra i nodi devono
essere allineate rapidamente. Inoltre questa struttura permette di
inviare il modello in modalità streaming non richiedendo di tenere tutto
il modello in memoria, questo aiuta specialmente lato server perché
permette di aumentare il numero di client concorrenti che possono essere
gestiti.

Il formato è progettato per facilitare sia la compressione che la
quantizzazione dei dati, due tecniche essenziali per ridurre al minimo
il consumo della banda di rete. L’intestazione include un campo chiamato
diffed_model_version, che indica la versione globale del modello da cui
deriva la versione corrente. Grazie a questa funzionalità, i pesi del
modello possono essere salvati come differenza (o "diff") rispetto a una
versione di riferimento, anziché come valori assoluti. Questo approccio
riduce drasticamente la quantità di dati, poiché molte differenze
risultano vicine a zero, rendendo il file facilmente comprimibile con
algoritmi di compressione senza perdita.

Questa struttura di compressione incrementale, unita alla possibilità di
quantizzare i pesi (ad esempio, utilizzando una precisione inferiore
come int8 anziché float32), consente di ottimizzare ulteriormente la
trasmissione del modello nel contesto dell’apprendimento federato,
riducendo i requisiti di rete senza sacrificare l’accuratezza
complessiva del modello.

<figure>
<img src="assets/file-system.png" id="fig:file-system"
style="width:10cm" alt="Approccio di Client basato su file system" />
<figcaption aria-hidden="true">Approccio di Client basato su file
system</figcaption>
</figure>

## Architettura Software del Client

L’architettura software del client è estremamente semplice, consiste in
una libreria che nasconde l’interazione con il server centrale. La sua
interfaccia è il più minimale possibile, come si vede nel codice
sottostante, basta importare la libreria e definire un Client.

``` python
token = "auth_token..."
store = BlobStore()
client = FLClient("coordinator.domain", store, token, backend="pythorch")
net = client.get_model()

client.train(rounds=N, dataloader)
```

<span id="code:client" label="code:client"></span>

Dopodiché sono disponibile due funzioni: get_model permette di scaricare
il modello globale e di inizializzare la rete nel backend selezionato;
train permette di eseguire automaticamente N round di apprendimento
federato. La struttura del modello e l’implementazione della strategia
di training sono definite globalmente e fornite dal server centrale.
Queste funzioni sono personalizzabili in fase di configurazione del
server ma l’idea è quella di fornire automaticamente le funzioni
relative alle ottimizzazioni più comuni come FedAvg, FedProx, FedNova
etc. È comunque possibile riscrivere completamente la funzione di
training ed usare in modo esplicito le funzioni di ricezione eventi ed
invio dei modelli.

L’idea per il futuro è quella di rendere ancora più trasparente ed
agnostico il framework. Un modo per farlo potrebbe essere quello di
creare una soluzione che non dipenda dalla tecnologia implementativa
dell’applicazione. Per far questo si potrebbe creare un deamon che legga
i dati da una cartella del file system ed in background esegua
l’apprendimento federato. Inoltre lo stesso deamon avrebbe il compito di
mantenere aggiornato il modello globale utilizzato. In
Figura <a href="#fig:file-system" data-reference-type="ref"
data-reference="fig:file-system">4.5</a> è rappresentata l’architettura
ad alto livello della proposta. Con questa soluzione sviluppare
applicazioni che facciano uso di federated learning sarebbe molto
semplice, in quanto l’applicazione dove solo salvare i dati raccolti in
una cartella.

# Valutazione sperimentale

Prima di procedere con l’implementazione del framework descritto in
questa tesi, abbiamo condotto una serie di esperimenti preliminari per
valutare la fattibilità empirica del nostro approccio. Come approfondito
nel Capitolo <a href="#chap:architettura" data-reference-type="ref"
data-reference="chap:architettura">4</a>, il framework richiede
l’adozione di un approccio asincrono per soddisfare i requisiti di un
sistema caratterizzato da forte eterogeneità e da comunicazioni non
pienamente affidabili. Tuttavia, come discusso nella
sezione <a href="#sub:sync-async" data-reference-type="ref"
data-reference="sub:sync-async">2.6</a>, l’adozione di un approccio
asincrono introduce numerose sfide legate alla convergenza e alla
stabilità del modello, problematiche che possono essere ulteriormente
aggravate in presenza di configurazioni eterogenee dei Client.

Per valutare l’impatto dell’asincronia sul sistema, abbiamo condotto
esperimenti iniziali in contesti sincroni o ibridi, introducendo
progressivamente assunzioni più rilassate. Questo approccio graduale ci
ha permesso di comprendere in maniera più approfondita le criticità
derivanti dall’eterogeneità, offrendo una base solida per affrontare le
problematiche che emergono in ambienti di esecuzione asincroni.

## Architettura dei test

Per valutare l’approccio proposto, abbiamo progettato un’architettura di
test su un cluster di 16 nodi omogenei per condurre esperimenti in un
contesto distribuito e con numerosi Client. Questa scelta differisce
significativamente da altri framework di simulazione dell’apprendimento
federato, come Flower, che tipicamente eseguono le simulazioni su una
singola macchina utilizzando processi o thread separati per
rappresentare i Client. La nostra soluzione è progettata per replicare
in modo realistico le condizioni operative di un sistema distribuito,
offrendo quindi una maggiore fedeltà a scenari realistici e maggiore
scalabilità orizzontale.

#### Architettura del Cluster

L’architettura distribuita è composta da:

-   Un nodo dedicato al Server Ticker (le cui funzionalità verranno
    introdotte nella prossima sezione) ed all’Aggregatore, per il
    coordinamento dei Client e la gestione asincrona degli
    aggiornamenti.

-   Quindici nodi Client, ciascuno rappresentante un’entità indipendente
    del sistema federato, che esegue il ciclo di allenamento locale ed
    effettua la comunicazione con l’Aggregatore.

Ogni nodo dispone di una CPU a 8 core con hyperthreading e memoria
sufficiente per eseguire i carichi di lavoro assegnati. La comunicazione
tra i nodi avviene tramite una rete locale ad alta velocità di tipo
Ethernet. L’architettura distribuita proposta offre una scalabilità
nativa, rendendo il framework adatto per esperimenti di dimensioni
crescenti senza richiedere modifiche strutturali. In particolare:

È possibile aggiungere nuovi nodi Client al cluster con un impatto
minimo sull’ infrastruttura esistente. L’utilizzo di nodi fisici
permette di sfruttare pienamente le risorse hardware, migliorando le
prestazioni e riducendo il rischio di generare colli di bottiglia dovuti
all’esecuzione concorrente di numerosi Client su un solo server. Grazie
alla scelta di eseguire i test su un cluster distribuito, il nostro
framework rappresenta un approccio più realistico e scalabile rispetto
alle soluzioni basate su simulazioni centralizzate, dimostrando la sua
capacità di operare efficacemente in ambienti complessi e su larga
scala.

#### Componenti dell’Architettura

Il sistema è suddiviso in tre componenti principali: il Server Ticker, i
Client ed un Aggregatore (Sever del framework ampiamente descritto nel
Capitolo <a href="#chap:architettura" data-reference-type="ref"
data-reference="chap:architettura">4</a>) che gestisce l’apprendimento
federato in modalità completamente asincrona.

Componenti principali:

-   Server Ticker: ha il compito di sincronizzare il comportamento dei
    Client, senza partecipare direttamente al processo di aggregazione.
    Le sue funzioni principali sono:

    -   Invio di Tick: invia segnali periodici (tick) a tutti i Client
        connessi, indicando l’inizio di un nuovo ciclo di computazione.

    -   Sincronizzazione dei Client: attende gli ack dai Client, che
        confermano la ricezione del tick. Prima di inviare un nuovo
        tick, il Server Ticker aspetta che i Client abbiano completato
        il ciclo corrente o abbiano gestito eventuali ritardi.

    Il Server Ticker non interagisce con il modello globale né raccoglie
    aggiornamenti: la sua funzione è esclusivamente quella di coordinare
    i Client scandendo il tempo.

-   Client: Ogni nodo del cluster esegue una o più istanze di Client,
    che hanno il compito di: a) ricevere il modello globale
    dall’Aggregatore; b) eseguire il ciclo di allenamento locale; c)
    inviare gli aggiornamenti del modello all’Aggregatore.

    Il comportamento sincrono o asincrono dipende direttamente dai
    Client. Quest’ultimi possono:

    1.  inviare immediatamente un ack al tick ricevuto, consentendo al
        Server Ticker di proseguire con il prossimo ciclo (comportamento
        asincrono).

    2.  ritardare l’invio dell’ack, introducendo uno stallo controllato
        nel sistema o una sua parte e quindi forzando la
        sincronizzazione di solo alcune fasi dell’apprendimento
        federato.

-   Aggregatore: componente completemente asincrono che gestisce
    esclusivamente la ricezione degli aggiornamenti dai Client,
    l’aggregazione e la distribuzione dei modelli globali. Non vi è
    alcun coordinamento diretto tra Aggregatore e Server Ticker dato che
    la sincronizzazione tra i Client è gestita esclusivamente dal
    Ticker.

L’architettura di test è mostrata ad alto livello della
Figura <a href="#fig:test-arch" data-reference-type="ref"
data-reference="fig:test-arch">5.1</a>

<figure>
<img src="assets/test-arch.png" id="fig:test-arch" style="width:100.0%"
alt="Architettura dei test" />
<figcaption aria-hidden="true">Architettura dei test</figcaption>
</figure>

## Problemi di eterogeneità in contesti sincroni

Nella maggior parte dei paper di ricerca si affronta il tema dell’
eterogeneità dei sistemi di calcolo per quanto riguarda Federated
Average (FedAvg) semplicemente ignorando gli aggiornamenti in ritardo.
Per aggiornamenti in ritardo si intendono gli aggiornamenti che sono
iniziati da una versione del modello globale che non è più l’ultima al
momento del loro invio. I Client che inviano aggiornamenti in ritardo
vengono detti stragglers. Essendo l’intento finale quello di sviluppare
un sistema asincrono volevamo capire l’impatto sulla convergenza nel
caso in cui si considerino comunque tali aggiornamenti.

Per fare ciò in questa sezione studiamo l’impatto della presenza di
ritardi e dell’ aggregazione di aggiornamenti obsoleti aggiungendo in
modo incrementale problematiche di eterogeneità (sia statistica, che di
sistema). Per confermare i risultati degli esperimenti ognuno di essi è
stato ripetuto 8 volte; per praticità e compattezza di seguito verranno
riportati i grafici di uno solo di essi.

#### Pattern di ritardo

Per rendere gli esperimenti discussi successivamente deterministici e
quindi replicabili, abbiamo definito 3 pattern principali di ritardo:

-   **Campionamento**: Consideriamo *N*<sub>*t*</sub> la quantità di
    Client attivi al round *t*, ⌈*N*<sub>*t*</sub> \* *p*⌉ è il numero
    di Client stragglers in quel round *t*.

-   **Latenza**: Questo pattern utilizza due parametri *L*, *p*: *L*
    descrive la quantità di ritardo che un client straggler ha rispetto
    al modello globale. Per esempio *L* = 2 implica che un Client
    straggler invia un aggiornamento relativo al modello globale vecchio
    ormai di due generazioni. *p* descrive la probabilità che un Client
    sia uno straggler. Per esempio se ci sono 100 Client e *p* = 0.5, 50
    Client sono stragglers. Questo pattern ci aiuta a capire come varia
    la stabilità della convergenza del modello nel caso in cui ci siano
    tipologie diverse di Client che hanno tempi diversi di elaborazione
    durante l’intero apprendimento federato.

-   **Random**: Questo pattern fa uso di un generatore di numeri pseudo
    casuali. Ad ogni round ogni client ha una probabilità *p* di essere
    in ritardo. Il numero di generazioni di ritardo è dunque dato dalla
    distribuzione geometrica di parametro 1 − *p*.

#### Stragglers con dati IID

Per gettare le basi ed un riferimento di base per le sezioni successive
abbiamo iniziato con il caso base: apprendimento sincrono su dati IID.
L’approccio utilizzato non è teoricamente totalmente sincrono ma lo è in
pratica: anche se vengono permessi aggiornamenti relativi a vecchie
versioni del modello globale. Infatti non viene prodotto un modello
globale ogni qual volta un aggiornamento viene ricevuto ma, quando tutti
gli aggiornamenti dei Client non stragglers vengono ricevuti.

Per i primi test abbiamo utilizzato i dataset MNIST (ML di
riconoscimento di immagini) ed AGNews (ML di classificazione del testo).
Abbiamo utilizzato AGNews per verificare l’indipendenza dei risultati
dal tipo di ML e di dataset. Per compattezza d’ora in avanti verrà
mostrato solamente il caso di MNIST (AGNews ha risultati che confermano
lo stesso andamento). I dataset sono stati divisi uniformemente su *N*
Clients. Per uniformemente si intende che ogni client ha lo stesso
numero di campioni per ogni classe. Per notare esclusivamente l’effetto
degli stragglers ed isolare la variabile di eterogeneità di sistema
abbiamo utilizzato gli stessi parametri di apprendimento per ogni client
(Epochs=6, BatchSize=256, LearningRate=0.01).

Essendo lo scopo della tesi quello di proporre un framework in grado di
gestire un grande numero di client, abbiamo utilizzato 100 Client, a
differenza di studi condotto in altre ricerche come in cui viene
utilizzato un numero limitato di Client (nel range di poche decine:
*C**l**i**e**n**t* ∈ \[10,30\]).

Abbiamo ripetuto l’esperimento variando i pattern di ritardo. In
Tabella <a href="#tab:test-iid1-p" data-reference-type="ref"
data-reference="tab:test-iid1-p">5.1</a> sono mostrati i parametri dei
test eseguiti con il pattern di ritardo Latenza (descritto nella sezione
precedente). Oltre al pattern Latenza abbiamo eseguito i test con il
pattern Random seguendo le stesse probabilità *p*.

<div id="tab:test-iid1-p">

| L/p |    0     |    0.1     |    0.2     |    0.4     |    0.6     |
|:---:|:--------:|:----------:|:----------:|:----------:|:----------:|
|  0  | L 0, p 0 | L 0, p 0.1 | L 0, p 0.2 | L 0, p 0.4 | L 0, p 0.6 |
|  2  | L 2, p 0 | L 2, p 0.1 | L 2, p 0.2 | L 2, p 0.4 | L 2, p 0.6 |
|  4  | L 4, p 0 | L 4, p 0.1 | L 4, p 0.2 | L 4, p 0.4 | L 4, p 0.6 |
|  8  | L 8, p 0 | L 8, p 0.1 | L 8, p 0.2 | L 8, p 0.4 | L 8, p 0.6 |

Parametri ritardo di tipo Latenza test su dati IID

</div>

Essendo i risultati molto omogenei tra di loro, in
Figura <a href="#fig:t00_MNIST_stag_even" data-reference-type="ref"
data-reference="fig:t00_MNIST_stag_even">5.2</a> sono presentati i
risultati più rilevanti. Dai risultati si deduce il fatto che permettere
l’aggregazione di aggiornamenti obsoleti con dati i.i.d non inficia
sull’accuratezza del modello finale. Già dal round 50 il delta è così
vicino a 0 che diventa trascurabile. Per delta si intende la differenza
tra l’ accuracy del modello allenato in un contesto privo di ritardi e
l’ accuracy del modello allenato con un pattern di ritardo specifico.

<figure>
<img src="assets/test00_MNIST_strugglers_even2.png"
id="fig:t00_MNIST_stag_even" style="width:65.0%"
alt="Delta accuracy del modello al proseguire dei round di allenamento rispetto ad un allenamento senza stragglers con dati i.i.d. delta(round) = accuracy_{no\_strag}(round) - accuracy_{strag}(round)" />
<figcaption aria-hidden="true">Delta accuracy del modello al proseguire
dei round di allenamento rispetto ad un allenamento senza stragglers con
dati i.i.d.<br />
<span
class="math display"><em>d</em><em>e</em><em>l</em><em>t</em><em>a</em>(<em>r</em><em>o</em><em>u</em><em>n</em><em>d</em>) = <em>a</em><em>c</em><em>c</em><em>u</em><em>r</em><em>a</em><em>c</em><em>y</em><sub><em>n</em><em>o</em>_<em>s</em><em>t</em><em>r</em><em>a</em><em>g</em></sub>(<em>r</em><em>o</em><em>u</em><em>n</em><em>d</em>) − <em>a</em><em>c</em><em>c</em><em>u</em><em>r</em><em>a</em><em>c</em><em>y</em><sub><em>s</em><em>t</em><em>r</em><em>a</em><em>g</em></sub>(<em>r</em><em>o</em><em>u</em><em>n</em><em>d</em>)</span></figcaption>
</figure>

#### Stragglers con dati non uniformemente distribuiti

Cosa accade però se la distribuzione relativa al tipo di dati non è
uniforme? Abbiamo ripetuto l’esperimento con una distribuzione non
uniforme delle labels in ogni dataset locale (la distribuzione delle
labels è mostrata in
Figura <a href="#fig:uneven_distr" data-reference-type="ref"
data-reference="fig:uneven_distr">[fig:uneven_distr]</a>). La
distribuzione dei campioni, come di vede, è molto verticale su una
singola classe, volevamo sperimentare il caso limite per vedere quanto
questa variabile possa potenzialmente incidere sulla convergenza del
modello al variare del numero di stragglers. Come si vede nella
Figura <a href="#fig:t00_acc" data-reference-type="ref"
data-reference="fig:t00_acc">5.3</a> la convergenza finale non risente
della distribuzione non uniforme della tipologia di campioni nei dataset
locali. Nella
Figura <a href="#fig:t00_MNIST_strugglers_uneven" data-reference-type="ref"
data-reference="fig:t00_MNIST_strugglers_uneven">5.4</a> si vede inoltre
che la presenza di strugglers non causa grossi problemi o meglio, non ne
causa affatto, infatti gli esperimenti eseguiti con i vari pattern di
ritardo mostrano con una quantità ragionevole di scarto lo stesso
andamento del caso con nessun ritardo.

<figure>
<img src="assets/test00_accuracy2.png" id="fig:t00_acc"
style="width:60.0%"
alt="Accuracy dei modelli allenati in assenza di ritardi. Come si nota la distribuzione delle label non inficia né sulle prestazioni del modello finale né sulla velocità di convergenza." />
<figcaption aria-hidden="true">Accuracy dei modelli allenati in assenza
di ritardi. Come si nota la distribuzione delle label non inficia né
sulle prestazioni del modello finale né sulla velocità di
convergenza.</figcaption>
</figure>

<figure>
<img src="assets/test00_MNIST_strugglers_uneven.png"
id="fig:t00_MNIST_strugglers_uneven" style="width:65.0%"
alt="Delta accuracy del modello al proseguire dei round di allenamento rispetto ad un allenamento senza stragglers con dati non uniformemente distribuiti. delta(round) = accuracy_{no\_strag}(round) - accuracy_{strag}(round)" />
<figcaption aria-hidden="true">Delta accuracy del modello al proseguire
dei round di allenamento rispetto ad un allenamento senza stragglers con
dati non uniformemente distribuiti. <span
class="math display"><em>d</em><em>e</em><em>l</em><em>t</em><em>a</em>(<em>r</em><em>o</em><em>u</em><em>n</em><em>d</em>) = <em>a</em><em>c</em><em>c</em><em>u</em><em>r</em><em>a</em><em>c</em><em>y</em><sub><em>n</em><em>o</em>_<em>s</em><em>t</em><em>r</em><em>a</em><em>g</em></sub>(<em>r</em><em>o</em><em>u</em><em>n</em><em>d</em>) − <em>a</em><em>c</em><em>c</em><em>u</em><em>r</em><em>a</em><em>c</em><em>y</em><sub><em>s</em><em>t</em><em>r</em><em>a</em><em>g</em></sub>(<em>r</em><em>o</em><em>u</em><em>n</em><em>d</em>)</span></figcaption>
</figure>

#### Stragglers in caso non i.i.d

Fino ad ora abbiamo testato solamente un caso di eterogeneità
statistica: la distribuzione dei dati. Cosa succede se i modelli locali
non sono gli stessi, ovvero, se non esiste una funzione obiettivo
globale univoca?

Per testare come cambia la convergenza quando i modelli locali sono
eterogenei tra di loro, abbiamo replicato gli esperimenti proposti in .
Questi esperimenti prevedono l’uso del Federated Learning per allenare
una regressione logistica (Logistic Regression) su dataset sintetici.

I dataset vengono generati a partire da due parametri *α* e *β* dove il
primo controlla quanto i modelli locali siano diversi tra di loro ed il
secondo quanto i dati sono distribuiti in modo non uniforme (viene
aggiunto il caso i.i.d come benchmark di riferimento).

Per generare questi dataset ci siamo avvalsi dell’artefatto software
fornito dagli autori [^3], utilizzando gli stessi seed per i generatori
numerici pseudo casuali in modo da ottenere esattamente gli stessi
dataset utilizzati.

Per prima cosa come benchmark eseguiamo il test rimuovendo i ritardi e
quindi la presenza di Client stragglers. Dalla
Figura <a href="#fig:non-iid" data-reference-type="ref"
data-reference="fig:non-iid">[fig:non-iid]</a> Si nota in modo evidente
quanto la stabilità del modello globale sia compromessa nei casi non
i.i.d e come sia proporzionalmente instabile all’aumentare
dell’eterogeneità statistica. Basta però utilizzare parametri più
ragionevoli come un batch size più grande o un learning rate più basso
per riportare la stabilità del modello. Un ulteriore opzione è quella di
utilizzare ottimizzazioni del classico algoritmo FedAvg come FedProx che
risolve i problemi di instabilità come mostrato nell’articolo citato.
Queste soluzioni sono efficienti e permettono di poter assicurare in
pratica la convergenza del modello globale anche in situazioni
eterogenee.  

Cosa succede se reintroduciamo la presenza degli stragglers? Abbiamo
ripetuto gli esperimenti in due configurazioni:

-   Scarto di aggiornamenti obsoleti: Ogni qualvolta che un Client è in
    ritardo con il proprio aggiornamento il risultato del suo
    allenamento viene scartato. Per quanto riguarda il pattern di
    ritardo questi esperimenti utilizzano il Campionamento. I risultati
    sono mostrati in
    Figura <a href="#fig:non-iid-strugglers" data-reference-type="ref"
    data-reference="fig:non-iid-strugglers">[fig:non-iid-strugglers]</a>

-   Aggregazione degli aggiornamenti obsoleti: Quando arriva un
    aggiornamento di un Client in ritardo si aggrega il risultato
    obsoleto insieme ad i risultati in linea con il modello globale. Per
    quanto riguarda il pattern di ritardo questi esperimenti utilizzano
    lo schema Random. In
    Figura <a href="#fig:non-iid-strugglers-nodrop" data-reference-type="ref"
    data-reference="fig:non-iid-strugglers-nodrop">[fig:non-iid-strugglers-nodrop]</a>
    vengono mostrati i risultati di FedAvg mentre in
    Figura <a href="#fig:non-iid-fedprox" data-reference-type="ref"
    data-reference="fig:non-iid-fedprox">[fig:non-iid-fedprox]</a>
    quelli di FedProx.

Nel caso di scarto degli aggiornamenti obsoleti si nota una maggiore
instabilità della convergenza del modello globale rispetto all’assenza
di stragglers (Figura <a href="#fig:non-iid" data-reference-type="ref"
data-reference="fig:non-iid">[fig:non-iid]</a>), ma utilizzando
parametri di allenamento più moderati come un learning rate più basso si
risolve il problema. Nel caso di aggregazione degli aggiornamenti
obsoleti, invece, si ha una maggior instabilità rispetto alla tecnica
dello scarto; in questo caso adattare i parametri di addestramento non
risulta sufficiente, mentre utilizzare tecniche come FedProx risolve il
problema di instabilità. Infatti come mostrato in
Figura <a href="#fig:non-iid-fedprox" data-reference-type="ref"
data-reference="fig:non-iid-fedprox">[fig:non-iid-fedprox]</a> si nota
quanto FedProx sia efficace in questi casi, rimanendo stabile in casi di
parametrizzazioni non ottimali. Si noti inoltre come il caso i.i.d
rallenti notevolmente la convergenza in caso di forte presenza di
stragglers e in caso di utilizzo di learning rate molto limitati.

#### Stragglers con parametri di allenamento eterogenei

Fino ad ora abbiamo posto l’assunto che ogni Client esegua la stessa
quantità di lavoro globale, ed abbiamo notato che in questa
configurazione, sebbene la presenza di stragglers infici sulla stabilità
del modello, questa è facilmente gestibile con piccoli accorgimenti
nella configurazione dei parametri di apprendimento.

Cosa succede se rilassiamo questa assunzione e permettiamo ad i Client
di eseguire una quantità variabile di lavoro?

Per permettere questo esperimento abbiamo rimosso gli stragglers ed
abbiamo permesso ad ogni Client di eseguire una quantità variabile di
lavoro in termine di epoche. Così facendo abbiamo rimosso il fenomeno
degli stragglers permettendo a questi Client di eseguire meno lavoro per
non essere in ritardo. Come mostrato nell’articolo , e come replicato
con i nostri test in
Figura <a href="#fig:var_ephocs" data-reference-type="ref"
data-reference="fig:var_ephocs">[fig:var_ephocs]</a> FedAvg in
condizione non i.i.d tende a perdere stabilità nella convergenza e
presenta il problema del’ inconsistenza dell’obiettivo spigato in
dettaglio nella sezione
Eterogeneità dei Sistemi di Calcolo <a href="#chap:sys-eter" data-reference-type="ref"
data-reference="chap:sys-eter">2.3</a>. FedProx in questo caso però,
aggiungendo un parametro di prossimità e quindi limitando gli effetti di
aggiornamenti che seguono direzioni diverse, riesce a mascherare
efficacemente il problema dell’eterogeneità di sistema. FedProx non è
l’unica ottimizzazione possibile infatti esistono soluzioni come
FedNova , FedAvgM  ed HySync  che utilizzano approcci differenti ma
mirati al miglioramento della stabilità della convergenza del modello e
in modo specifico alla mitigazione del problema di inconsistenza
dell’obiettivo.

  

## Problemi di eterogeneità in contesti asincroni

Avendo approfondito gli effetti dell’eterogeneità in contesti sincroni,
abbiamo acquisito le conoscenze e i benchmark necessari per valutare in
maniera sistematica l’approccio asincrono. Il passo successivo consiste
nel definire un metodo che consenta di simulare l’esecuzione asincrona
dei Client in un ambiente controllato e ripetibile. Questo approccio è
fondamentale per analizzare le dinamiche asincrone, identificare
eventuali criticità e ottimizzare le strategie di aggregazione e
gestione dei Client, tenendo conto delle loro caratteristiche
eterogenee.

#### Modellazione dell’aggiornamento asincrono e selezione dei Client

Per ottenere un comportamento realistico dell’ invio degli aggiornamenti
abbiamo utilizzato un generatore numerico pseudo casuale con un seme
costante. Essendo l’apprendimento totalmente asincrono il concetto di
round decade. Definiamo l’iterazione della nostra simulazione *tick*.
Ogni tick rappresenta in maniera astratta un lasso di tempo variabile,
poiché simula il passare del tempo tra l’invio dei modelli da parte dei
client e questo è arbitrariamente non uniforme. Allo scopo dei test gli
intervalli di tempo che occorrono tra un aggiornamento e l’ altro non ci
interessano. Poiché non esiste più il concetto di round sull’asse delle
*x* indichiamo le versioni del modello globale (indicate da 0 a *n*:
dalla più vecchia alla più nuova).

Ad ogni tick viene selezionato in modo casuale un Client che deve
inviare un aggiornamento. Il Client selezionato esegue *e* epoche di
allenamento che, per lo scopo di questo esperimento, sono un numero
casuale tra (\[5,20\])[^4]. Inoltre il client selezionato allena il
modello basato su *l* versioni precedenti del modello globale (dove *l*
può essere 0, quindi l’ ultimo modello), in modo da simulare l’effetto
degli aggiornamenti obsoleti. Il parametro *l* è legato ad una
distribuzione geometrica di parametro *p* (distribuzione descritta dalla
formula *P*(*l*=*n*) = *p*<sup>*n*</sup>). Per l’esecuzione di questo
esperimento abbiamo fissato *p* = 0.8 in quanto una probabilità di
ritardo molto alta che ci permette di studiare il caso più estremo.

Per motivi di complessità computazionale e necessità di confronto con la
fase precedente abbiamo utilizzato i dataset sintetici generati come
descritto nella sezione antecedente. L’utilizzo di questi dataset
sintetici ci permette inoltre di avere il controllo sulle variabili di
eterogeneità statistica.

#### Problemi di instabilità e soluzioni

Come primo esperimento abbiamo valutato il caso di aggregazione
puramente asincrona: Ogni volta che viene ricevuto un aggiornamento da
parte dell’ aggregatore questo genera subito il nuovo modello globale.
In Figura <a href="#fig:pure-async" data-reference-type="ref"
data-reference="fig:pure-async">5.5</a> sono mostrati i risultati. Come
si nota in caso di dati i.i.d non si riscontrano problemi ma, nel
momento in cui si introduce un margine di eterogeneità la convergenza
del modello diventa totalmente instabile.

<figure>
<img src="assets/async_e=(8,20)_p=0.8_bs=50_lr=0.01.png"
id="fig:pure-async" style="width:80.0%"
alt="Accuracy allenamento totalmente asincrono." />
<figcaption aria-hidden="true">Accuracy allenamento totalmente
asincrono.</figcaption>
</figure>

Per compensare questa instabilità abbiamo introdotto la tecnica di
FedBuff , la quale consiste nel "bufferizzare" gli aggiornamenti. Appena
arriva un aggiornamento lo si mette nel buffer, se ci sono almeno *N*
aggiornamenti pendenti si genera il nuovo modello altrimenti si aspetta
per nuovi aggiornamenti. Il framework proposto supporta questa
operazione nativamente, infatti, per motivi di performance, gli
aggiornamenti vengono automaticamente messi in una coda. Abbiamo dunque
ripetuto l’esperimento applicando FedBuff con *N* = 10. Come si vede in
Figura <a href="#fig:fed-buff" data-reference-type="ref"
data-reference="fig:fed-buff">5.6</a>, dove sono mostrati i risultati,
questa ottimizzazione è decisamente efficiente e permette di risolvere i
problemi di instabilità. Nel caso di maggior eterogeneità
(*α* = 1, *β* = 1) si notato dei picchi negativi sull’ accuracy del
modello, questi sono probabilmente legati problema dell’inconsistenza
dell’obiettivo: ad una nuova versione collaborano modelli che presentano
tutti un forte "bias", questo fa si che il modello globale si sposti
verso il "bias" e che quindi si allontani dalla funzione obiettivo
globale.

<figure>
<img src="assets/fedbuff(10)_e=(8,20)_p=0.8_bs=50_lr=0.01.png"
id="fig:fed-buff" style="width:80.0%"
alt="Accuracy allenamento asincrono con FedBuff N=10" />
<figcaption aria-hidden="true">Accuracy allenamento asincrono con
FedBuff <span class="math inline"><em>N</em> = 10</span></figcaption>
</figure>

#### Combinazione di FedBuff e FedProx 

Sebbene FedBuff sia già potenzialmente sufficiente per risolvere i
problemi di instabilità, è possibile combinare questa tecnica con altre
tecniche come FedProx. Quindi abbiamo ripetuto l’esperimento mantenendo
*N* = 10 ed integrando FedProx con parametro *m**u* = 1. Come si vede
dalla Figura <a href="#fig:fed-buff2" data-reference-type="ref"
data-reference="fig:fed-buff2">5.7</a> le piccole instabilità che si
erano ottenute con l’applicazione del semplice FedBuff sono sparite.
Come effetto collaterale però abbiamo un notevole rallentamento della
convergenza in caso di dati i.i.d, questo risultato è in linea con le
aspettative infatti questo fenomeno è evidenziato anche nell’articolo in
cui viene proposto FedProx 

<figure>
<img src="assets/fedbuff(10)_mu=1_e=(8,20)_p=0.8_bs=50_lr=0.01.png"
id="fig:fed-buff2" style="width:80.0%"
alt="Accuracy allenamento asincrono con FedBuff N=10 e FedProx mu=1" />
<figcaption aria-hidden="true">Accuracy allenamento asincrono con
FedBuff <span class="math inline"><em>N</em> = 10</span> e FedProx <span
class="math inline"><em>m</em><em>u</em> = 1</span></figcaption>
</figure>

## Framework benchmarks

Per valutare le prestazioni del framework implementato in questa tesi
abbiamo eseguito dei benchmark. Come metrica di riferimento abbiamo
usato il numero di operazioni al secondo (OPS) che il sistema riesce a
gestire per valutare le capacità di throughput del server. Abbiamo
creato due tipi di operazioni che i Client di test possono eseguire: 1)
la prima consiste nel connettersi al server, autenticarsi e scaricare
l’ultima versione del modello; 2) la seconda nel connettersi,
autenticarsi ed inviare un aggiornamento locale. Ogni Client eseguirà in
modo alternato le due operazioni. Per rendere il test rappresentativo di
un contesto di Federated Learning reale abbiamo ritenuto ragionevole
considerare che ogni dispositivo federato impieghi un tempo nell’ordine
di minuti per allenare il modello locale. È quindi molto probabile che
calcolare un nuovo modello ogni secondo o frazione di secondo non sia
utile ai fini dell’apprendimento federato. Per cui abbiamo configurato
il server centrale in modo che aggreghi i risultati ogni 10 secondi.

I test sono stati eseguiti con vari gradi di parallelismo per valutare
la scalabilità del framework: 1, 2, 4, 8 threads, tenendo conto che
l’aggregatore risiede sempre su un thread separato. Ogni test è stato
ripetuto 10 volte ed i risultati finali sono la media delle prove
ripetute.

<figure>
<img src="assets/bench.png" id="fig:bench" style="width:80.0%"
alt="Benchmark del framework in termini di operazioni al secondo gestibili al variare del grado di parallelismo. La linea indicata l’efficienza dell’utilizzo delle risorse espressa in percentuale, ad indicare quanto il sistema sia in grado di utilizzare le risorse disponibili al variare del numero di thread utilizzati." />
<figcaption aria-hidden="true">Benchmark del framework in termini di
operazioni al secondo gestibili al variare del grado di parallelismo. La
linea indicata l’efficienza dell’utilizzo delle risorse espressa in
percentuale, ad indicare quanto il sistema sia in grado di utilizzare le
risorse disponibili al variare del numero di thread
utilizzati.</figcaption>
</figure>

In Figura <a href="#fig:bench" data-reference-type="ref"
data-reference="fig:bench">5.8</a> vengono presentati i risultati
ottenuti. Come si nota già con 4 thread il sistema è in grado di gestire
circa 100.000 OPS. Se si suppone che un Client sia in grado di generare
un aggiornamento ogni ora, il sistema è in grado di gestire un numero di
dispositivi dell’ordine del milione. Le prestazioni del framework
smettono di scalare già con 4 threads. Questo comportamento è dovuto
alla saturazione delle risorse di rete del cluster su cui sono stati
eseguiti i test. Dal punto di vista teorico, il server è progettato per
scalare in modo lineare con il numero di thread, grazie al design
parallelizzato e al thread separato dedicato all’aggregazione. Tuttavia,
le limitazioni delle risorse hardware disponibili, in particolare quelle
legate alla larghezza di banda e alla latenza della rete, hanno
rappresentato un collo di bottiglia per le prestazioni complessive.
Pertanto, i risultati presentati vanno ritenuti parziali e
rappresentativi solo delle capacità del framework all’interno del
contesto specifico di test. Come sviluppo futuro, prevediamo di valutare
le prestazioni del framework su reti di interconnessione a più larga
banda.

## Considerazioni

Sebbene l’eterogeneità statistica e di sistema pongano delle
problematiche a livello di convergenza del modello, nella prima sezione
di sperimentazione abbiamo mostrato empiricamente come queste, in
contesti sincroni, sono facilmente gestibili utilizzando configurazioni
dei parametri di allenamento appropriate e tecniche di ottimizzazione
dove necessario.  
In contesti asincroni invece, rispetto a quelli sincroni, l’eterogeneità
statistica e di sistema creano problemi di instabilità maggiori.
Utilizzando un approccio completamente asincrono (Dove si genera una
nuova versione del modello globale ad ogni aggiornamento ricevuto)
abbiamo notato il modello divergere in pratica nei casi in cui venga
meno la premessa di dati i.i.d. Tuttavia abbiamo notato miglioramenti
significativi nell’utilizzare approcci ibridi come FedBuff che
permettono di stabilizzare notevolmente la convergenza del modello.

Nonostante i notevoli miglioramenti ottenuti con FedBuff in casi di alta
eterogeneità rimangono, anche se attenuati, problemi di stabilità del
modello legati al fenomeno dell’inconsistenza dell’ obiettivo
(problematica discussa in dettaglio nel capitolo di
Background <a href="#background" data-reference-type="ref"
data-reference="background">2</a>).

Per risolvere il problema dell’instabilità del modello in contesti
altamente eterogenei abbiamo combinato FedBuff con FedProx . Questa
combinazione ha permesso di eliminare completamente il problema
dell’inconsistenza dell’obiettivo.  
Il lavoro svolto dimostra che, sebbene la gestione dell’eterogeneità in
contesti sincroni e asincroni possa sembrare una sfida complessa,
l’adozione di strategie appropriate consente di affrontare e risolvere
efficacemente questi problemi, garantendo la stabilità e la convergenza
del modello globale.  
Oltre ad i risultati relativi alla convergenza del modello globale
questi esperimenti hanno dimostrato che il framework[^5] sviluppato si è
dimostrato in grado di riprodurre risultati presentati in articoli
scientifici utilizzando simulatori, come e , in ambienti reali e
fisicamente distribuiti. Il framework si è rivelato sufficientemente
flessibile per introdurre più tecniche e verificarne la bontà (anche in
contesti sincroni che rimangono fuori dal scopo principale del software
stesso). La sua architettura modulare ha facilitato l’integrazione delle
varie strategie, in particolare quella di FedBuff, rendendo la sua
implementazione semplice ed efficace. Infatti, la bufferizzazione degli
aggiornamenti è già una funzionalità nativa del framework, progettata
per ottimizzare le performance. Il framework oltre ad una notevole
flessibilità ha dimostrato di essere in grado di gestire grossi carichi
di lavoro: nell’ordine di centinaia di migliaia di OPS.

# Conclusioni e Sviluppi futuri

Negli ultimi anni, il Federated Learning (FL) ha mostrato un grande
potenziale per superare i limiti del Machine Learning tradizionale,
offrendo soluzioni innovative in termini di privacy dei dati e
scalabilità. Tuttavia, la sua applicazione pratica, specialmente in
contesti con dispositivi eterogenei, presenta sfide significative legate
alle diverse capacità dei dispositivi, alla complessità delle
comunicazioni e alla gestione distribuita. In questa tesi, abbiamo
affrontato tali problematiche proponendo un framework[^6] progettato per
semplificare l’ adozione del Federated Learning asincrono, consentendo
agli sviluppatori di concentrarsi su aspetti chiave come la definizione
dei modelli e l’ analisi dei dati, senza doversi occupare delle
complessità tecniche.

Nel corso di questo elaborato abbiamo dapprima introdotto il Federated
Learning, le sue sfide (come l’eterogeneità dei dati e dei sistemi di
calcolo) ed alcune tra le possibili architetture. Dopo aver dato un
quadro completo del background necessario, abbiamo analizzato le
soluzioni attualmente presenti sul mercato, valutandone punti di forza e
debolezza, ed evidenziato una mancanza di opzioni appropriate per
contesti cross-device ed altamente eterogenei su grossa scala (centinaia
di migliaia di Client). Dopo aver chiarito le motivazioni che hanno
guidato lo sviluppo del framework oggetto di questa tesi abbiamo
descritto il design e l’architettura software in dettaglio. Infine
abbiamo condotto un’ampia sperimentazione per validare il funzionamento
stesso del software, riproponendo con successo esperimenti presenti in
articoli accademici come . Abbiamo inoltre dimostrato empiricamente che,
sebbene la gestione dell’eterogeneità in contesti sincroni e asincroni
possa sembrare una sfida complessa, l’ adozione di strategie appropriate
consente di affrontare e risolvere efficacemente questi problemi,
garantendo la stabilità e la convergenza del modello globale. Il
framework sviluppato si distingue per la sua attenzione all’efficienza e
alla flessibilità. La scelta di ottimizzare le prestazioni, riducendo il
carico computazionale e migliorando la gestione delle risorse, consente
una scalabilità superiore rispetto a molte soluzioni esistenti che
abbiamo analizzato. Anche la comunicazione tra dispositivi è stata
ottimizzata per garantire una maggiore fluidità e reattività del
sistema, permettendo di gestire efficacemente numerosi dispositivi
connessi in simultanea. Inoltre, il framework offre possibilità di
personalizzazione, consentendo di adattare le strategie di aggregazione
alle esigenze specifiche degli sviluppatori.I test sperimentali hanno
dimostrato come questo framework renda triviale lo sviluppo di
applicazioni di Federated Learning asincrono, affrontando con successo
problemi di eterogeneità e scalabilità tipici dei dispositivi
distribuiti.

#### Sviluppi futuri

Il framework sviluppato offre numerose possibilità di evoluzione per
affrontare in modo ancora più efficace le sfide legate al Federated
Learning. Una direzione promettente è il passaggio verso un’architettura
più decentralizzata e resiliente, con l’integrazione di meccanismi per
migliorare l’affidabilità e la scalabilità del sistema. L’adozione di
tecniche avanzate per ottimizzare le comunicazioni e ridurre il consumo
di banda rappresenta un altro potenziale ambito di miglioramento.  
Un’ulteriore area di sviluppo riguarda l’integrazione di strumenti
avanzati per la protezione della privacy, come la crittografia avanzata
o la privacy differenziale, che offrono garanzie ancora più solide sul
trattamento sicuro dei dati. Inoltre, l’estensione del supporto a una
gamma più ampia di modelli di learning, inclusi modelli non
supervisionati e di apprendimento per rinforzo, potrebbe rendere il
framework ancora più versatile e applicabile in contesti diversi.  
Infine, migliorare l’ accessibilità attraverso documentazione esaustiva
e interfacce intuitive potrebbe ampliare il pubblico di potenziali
utenti. L’efficacia del framework potrebbe essere ulteriormente validata
attraverso test su larga scala in collaborazione con realtà industriali
o accademiche, per dimostrare la sua capacità di operare con un elevato
numero di dispositivi in ambienti reali.

#### Considerazioni finali

Ritengo che il framework proposto, disponile su github, rappresenti un
passo avanti verso un Federated Learning più accessibile anche agli
sviluppatori non esperti in infrastrutture e sistemi distribuiti. Nella
sua fase di design e sviluppo abbiamo affrontato con efficacia le sfide
legate all’eterogeneità e alla scalabilità, fornendo agli sviluppatori
una soluzione flessibile e intuitiva per implementare applicazioni che
rispettano la privacy e sfruttano al meglio le risorse di dispositivi
eterogenei. Con i potenziali sviluppi futuri descritti nella sezione
precedente, l’obiettivo è ampliare ulteriormente le capacità
tecnologiche e di utilizzo del framework sviluppato, rendendolo un
elemento importante per l’innovazione nell’ambito del Federated
Learning, promuovendo la collaborazione, la sicurezza e l’efficienza in
ambienti distribuiti sempre più complessi.

[^1]: https://www.gsma.com/r/somic/

[^2]: https://www.unipoltech.com/it/news/scenario-iot

[^3]: https://github.com/litian96/FedProx

[^4]: Epoche inferiori a 5 potrebbero risultare insufficienti a
    consentire al modello di apprendere in modo significativo dai dati
    locali del client, portando a un aggiornamento poco utile o
    addirittura negativo per il modello globale. D’altra parte, eseguire
    più di 20 epoche potrebbe non essere necessario, poiché
    l’apprendimento federato si basa su aggiornamenti frequenti da parte
    dei client, e un numero elevato di epoche per ogni aggiornamento
    potrebbe portare a un sovraccarico computazionale eccessivo, senza
    apportare miglioramenti sostanziali al modello globale.

[^5]: https://github.com/mamodev/Async-Federated-Learnig

[^6]: <span id="nt:framework" label="nt:framework"></span>Codice
    sorgente: https://github.com/mamodev/Async-Federated-Learnig
