import sys
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Subset


def get_network(args):
    """ return given network
    """

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)

    # Sanity check: Check to see whether shuffled_dataset works with default ordering of CIFAR-100
    # cl_ordering = [*range(50000)]
    # shuffled_dataset = Subset(cifar100_training, cl_ordering)

    # Incorporate horrible logic to check whether CL works
    idx2Scores = [0.007952944375574589, 0.011873229406774044, 0.004782300908118486, 0.015215634368360043, 0.02263648435473442, 0.0060918149538338184, 0.03294837847352028, 0.017685232684016228, 0.010768207721412182, 0.009620736353099346, 0.026365643367171288, 0.055341873317956924, 0.015909506008028984, 0.012796031311154366, 0.016887197270989418, 0.008660515770316124, 0.02846277691423893, 0.04427049309015274, 0.018105093389749527, 0.04771406948566437, 0.012417752295732498, 0.00501762330532074, 0.03554140031337738, 0.08070695400238037, 0.013826148584485054, 0.03307521715760231, 0.03699443116784096, 0.03545283153653145, 0.01671621762216091, 0.007353531662374735, 0.01974426954984665, 0.01778741553425789, 0.006641886197030544, 0.018193071708083153, 0.015902426093816757, 0.010251895524561405, 0.016624970361590385, 0.02122713066637516, 0.0820913314819336, 0.011914149858057499, 0.026042822748422623, 0.006016136612743139, 0.02308214083313942, 0.0062232050113379955, 0.028056997805833817, 0.007420731708407402, 0.01583200879395008, 0.012564478442072868, 0.016687707975506783, 0.04874018579721451, 0.0067296260967850685, 0.009989212267100811, 0.03183137625455856, 0.013221189379692078, 0.05135471746325493, 0.023732967674732208, 0.014301441609859467, 0.06086244434118271, 0.007579835597425699, 0.004585613496601582, 0.00486296322196722, 0.0046889265067875385, 0.03525321185588837, 0.010437524877488613, 0.005104255862534046, 0.02478647418320179, 0.04370446130633354, 0.055198799818754196, 0.03674377128481865, 0.021377773955464363, 0.004209458827972412, 0.009079270996153355, 0.020343953743577003, 0.01064341515302658, 0.00810916069895029, 0.014112071134150028, 0.00804328452795744, 0.06278551369905472, 0.02960192784667015, 0.009898632764816284, 0.01697852648794651, 0.009349565021693707, 0.011697119101881981, 0.010909264907240868, 0.004668629728257656, 0.00400120671838522, 0.01869300939142704, 0.027157528325915337, 0.016291694715619087, 0.004151000175625086, 0.012593848630785942, 0.008194584399461746, 0.010400379076600075, 0.012845206074416637, 0.026520829647779465, 0.040599383413791656, 0.018253864720463753, 0.10536094009876251, 0.022008029744029045, 0.014472284354269505, 0.008864020928740501, 0.005143661517649889, 0.03337198495864868, 0.030404038727283478, 0.02375493384897709, 0.04143849015235901, 0.019197730347514153, 0.010282506234943867, 0.012563624419271946, 0.025095852091908455, 0.12251435220241547, 0.003943308722227812, 0.006009027361869812, 0.01070121955126524, 0.023093661293387413, 0.014907732605934143, 0.004474396351724863, 0.0029769698157906532, 0.005349546670913696, 0.01256457157433033, 0.02843436412513256, 0.006654523313045502, 0.011462182737886906, 0.021305974572896957, 0.017440684139728546, 0.042657457292079926, 0.015379113145172596, 0.004318934865295887, 0.0070030419155955315, 0.010327114723622799, 0.013178027234971523, 0.026242421939969063, 0.004462272860109806, 0.1298697292804718, 0.004031038843095303, 0.06486321985721588, 0.003727381117641926, 0.005438526626676321, 0.032860852777957916, 0.011018828488886356, 0.006470309570431709, 0.005323376972228289, 0.013401959091424942, 0.01655719242990017, 0.012107189744710922, 0.017025474458932877, 0.032825861126184464, 0.017306802794337273, 0.057855501770973206, 0.016535615548491478, 0.006743777543306351, 0.010336137376725674, 0.05462360009551048, 0.009182790294289589, 0.02486458048224449, 0.003786415094509721, 0.031884074211120605, 0.017036838456988335, 0.05369427055120468, 0.027570996433496475, 0.028996868059039116, 0.0239090733230114, 0.01425275206565857, 0.0038681684527546167, 0.015329224057495594, 0.09890926629304886, 0.0072539509274065495, 0.023558279499411583, 0.02203536406159401, 0.0156645979732275, 0.006134985946118832, 0.021359065547585487, 0.008879016153514385, 0.006614143960177898, 0.003606776474043727, 0.00858940090984106, 0.009241489693522453, 0.0100092189386487, 0.019293274730443954, 0.008852883242070675, 0.026792902499437332, 0.05012473464012146, 0.027103682979941368, 0.1029951348900795, 0.030695796012878418, 0.03120250068604946, 0.0020412495359778404, 0.09831863641738892, 0.03852337226271629, 0.021201610565185547, 0.00722646526992321, 0.009376023896038532, 0.011232171207666397, 0.03243515267968178, 0.022739240899682045, 0.03307512402534485, 0.015852682292461395, 0.06539864093065262, 0.03689327836036682, 0.005635568406432867, 0.002916982164606452, 0.009197170846164227, 0.03624647483229637, 0.018303701654076576, 0.012753898277878761, 0.011188509874045849, 0.02765670232474804, 0.03559564799070358, 0.06321578472852707, 0.005045595578849316, 0.00866550114005804, 0.014078705571591854, 0.023752672597765923, 0.008329663425683975, 0.034934889525175095, 0.023733047768473625, 0.06601262837648392, 0.015679974108934402, 0.007601677440106869, 0.04549794644117355, 0.008380884304642677, 0.09904616326093674, 0.022803209722042084, 0.019390037283301353, 0.06131238490343094, 0.023455630987882614, 0.02440282329916954, 0.038822367787361145, 0.010451612062752247, 0.012255366891622543, 0.004321920219808817, 0.011637954972684383, 0.009001669473946095, 0.014273191802203655, 0.04256094992160797, 0.014042375609278679, 0.005971638951450586, 0.008064616471529007, 0.003253102535381913, 0.009228096343576908, 0.011133462190628052, 0.007729912176728249, 0.012851115316152573, 0.030734777450561523, 0.021646494045853615, 0.037475910037755966, 0.009725594893097878, 0.05627544969320297, 0.007922087796032429, 0.009817955084145069, 0.013764005154371262, 0.008131769485771656, 0.0044046854600310326, 0.010723062790930271, 0.003680933266878128, 0.0036256660241633654, 0.02489294297993183, 0.01323650311678648, 0.009202477522194386, 0.026567451655864716, 0.022939534857869148, 0.003038923954591155, 0.00796644389629364, 0.05365985631942749, 0.018156345933675766, 0.00568951852619648, 0.015601571649312973, 0.007200213149189949, 0.011733128689229488, 0.008192216977477074, 0.010040870867669582, 0.0076665631495416164, 0.04413573443889618, 0.0035035626497119665, 0.022565195336937904, 0.05625136196613312, 0.010729311034083366, 0.018371546640992165, 0.023813892155885696, 0.00832198653370142, 0.012043263763189316, 0.004324428271502256, 0.01012219488620758, 0.007447904907166958, 0.012379703111946583, 0.008269349113106728, 0.031111424788832664, 0.01876177079975605, 0.020729143172502518, 0.005549610126763582, 0.01359013095498085, 0.005475581623613834, 0.01608065888285637, 0.028363725170493126, 0.04778444021940231, 0.004660187289118767, 0.01309927273541689, 0.04595748707652092, 0.01623218134045601, 0.025357386097311974, 0.013131270185112953, 0.03404898941516876, 0.005856224335730076, 0.04769885167479515, 0.010587658733129501, 0.05947989597916603, 0.006531032733619213, 0.0042868722230196, 0.01459912396967411, 0.012185793370008469, 0.02300558239221573, 0.03148499131202698, 0.04740820825099945, 0.004438386298716068, 0.011427515186369419, 0.00885858479887247, 0.011756270192563534, 0.013760148547589779, 0.015193012543022633, 0.030286353081464767, 0.015005255118012428, 0.01790761575102806, 0.02237425372004509, 0.023651912808418274, 0.04167243465781212, 0.004487418569624424, 0.042343541979789734, 0.027721090242266655, 0.04375382885336876, 0.01839148811995983, 0.039065733551979065, 0.020142601802945137, 0.007840019650757313, 0.008596034720540047, 0.0029408682603389025, 0.013847820460796356, 0.00753676425665617, 0.00802953913807869, 0.019510062411427498, 0.008636289276182652, 0.013621841557323933, 0.005919658578932285, 0.026822932064533234, 0.006789651233702898, 0.007365299854427576, 0.01974954828619957, 0.07471244782209396, 0.03107326477766037, 0.11348579078912735, 0.008244836702942848, 0.008126968517899513, 0.003837489988654852, 0.018365338444709778, 0.008817599155008793, 0.025128064677119255, 0.0034378962591290474, 0.010914832353591919, 0.0306030735373497, 0.01077007595449686, 0.005689071491360664, 0.0074666147120296955, 0.0032293498516082764, 0.008135520853102207, 0.0053993589244782925, 0.00351465935818851, 0.03199664503335953, 0.014495034702122211, 0.03522053360939026, 0.0030337038915604353, 0.01748521998524666, 0.03912913799285889, 0.006678986828774214, 0.02151380106806755, 0.02434014156460762, 0.038625460118055344, 0.016053905710577965, 0.014043303206562996, 0.035298850387334824, 0.007265984546393156, 0.012708080932497978, 0.032774075865745544, 0.011666644364595413, 0.005574438255280256, 0.040296949446201324, 0.002768325386568904, 0.024820351973176003, 0.014963020570576191, 0.020327767357230186, 0.015121249482035637, 0.009380411356687546, 0.030780619010329247]
    ordering = np.argsort(idx2Scores).tolist()

    cl_list = []
    for i in ordering:
        start_idx = (i - 1) * 128 + 1
        for j in range(start_idx, start_idx + 128):
            cl_list.append(j)

    cl_list = cl_list[:50000]
    shuffled_dataset = Subset(cifar100_training, cl_list)

    cifar100_training_loader = DataLoader(shuffled_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
