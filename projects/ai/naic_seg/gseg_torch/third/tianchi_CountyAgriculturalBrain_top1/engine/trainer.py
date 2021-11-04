'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors : now more
Description : 
LastEditTime: 2019-09-06 16:29:31
'''
import logging
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, Timer,TerminateOnNan
from ignite.metrics import  Loss, RunningAverage,Accuracy
import re
from .inference import *
from importer import *


def do_train(cfg,model,train_loader,val_loader,optimizer,scheduler,loss_fn,metrics,image_3_dataloader=None,image_4_dataloader=None):

    device = cfg.MODEL.DEVICE if torch.cuda.is_available() else 'cpu'
    epochs = cfg.SOLVER.MAX_EPOCHS
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Trainer")
    logger.info("Start training")
    trainer = create_supervised_trainer(model.train(),optimizer,loss_fn,device=device)
    trainer.add_event_handler(Events.ITERATION_COMPLETED,TerminateOnNan())

    evaluator = create_supervised_evaluator(model.eval(),metrics={"pixel_error":metrics},device=device)
    # evaluator_trainer = create_supervised_evaluator(model.eval(),metrics={"pixel_error":(metrics)},device=device)
    timer = Timer(average=True)
    timer.attach(trainer,start=Events.EPOCH_STARTED,resume=Events.ITERATION_STARTED,pause=Events.ITERATION_COMPLETED,step=Events.ITERATION_COMPLETED)
    RunningAverage(output_transform=lambda x:x).attach(trainer,'avg_loss')

    # 每 log_period 轮迭代结束输出train_loss
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        len_train_loader = len(train_loader)
        log_period = int(cfg.LOG_PERIOD*len_train_loader)
        iter = (engine.state.iteration-1)%len_train_loader + 1 + engine.state.epoch*len_train_loader
        if iter % log_period == 0:
            iter = (engine.state.iteration-1)%len_train_loader + 1
            logger.info("Epoch[{}] Iteration[{}/{}] Loss {:.7f}".format(engine.state.epoch,iter,len_train_loader,engine.state.metrics['avg_loss']))
            
    @trainer.on(Events.EPOCH_COMPLETED)
    def save(engine):
        epoch = engine.state.epoch
        print("epoch: "+str(epoch))
        if epoch%1 == 0:
            model_name=os.path.join(cfg.OUTPUT.DIR_NAME+"model/","epoch_"+str(engine.state.epoch)+"_"+cfg.TAG+"_"+cfg.MODEL.NET_NAME+".pth")
            torch.save(model.module.state_dict(),model_name)

    # 每val_period轮迭代结束计算一次val_metric
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_val_metric(engine):
        len_train_loader = len(train_loader)
        iter = (engine.state.iteration-1)%len_train_loader + 1 + engine.state.epoch*len_train_loader
        val_period = int(cfg.VAL_PERIOD*len_train_loader)
        if iter % val_period == 0:
            pass
            # 打印输出
            # evaluator.run(val_loader)
            # metrics = evaluator.state.metrics
            # avg_loss = metrics["pixel_error"]
            # logger.info("Validation Result - Epoch: {} Avg Pixel Accuracy: {:.7f} ".format(engine.state.epoch,avg_loss))

            ######################
            # # 分别用ttaforward
            # cfg.TOOLS.image_n = 3
            # image_3_predict = tta_forward(cfg,image_3_dataloader,model.eval())
            # pil_image_3 = Image.fromarray(image_3_predict)
            # image_3_save_path = "iter_" + str(iter) + "_" + "image_3_predict.png"
            # pil_image_3.save(os.path.join(r"./output",image_3_save_path))
            # image_3_label_save_path = "iter_" + str(iter) + "_" + "vis_" + "image_3_predict.jpg"
            # source_image_3 = cv.imread("./output/source/image_3.png")
            # mask_3 = label_resize_vis(image_3_predict,source_image_3)
            # cv.imwrite(os.path.join(r"./output",image_3_label_save_path),mask_3)

            # cfg.TOOLS.image_n = 4
            # image_4_predict = tta_forward(cfg,image_4_dataloader,model.eval())
            # pil_image_4 = Image.fromarray(image_4_predict)
            # image_4_save_path = "iter_" + str(iter) + "_" + "image_4_predict.png"
            # pil_image_4.save(os.path.join(r"./output",image_4_save_path))
            # image_4_label_save_path = "iter_" + str(iter) + "_" + "vis_" + "image_4_predict.jpg"
            # source_image_4 = cv.imread("./output/source/image_4.png")
            # mask_4 = label_resize_vis(image_4_predict,source_image_4)
            # cv.imwrite(os.path.join(r"./output",image_4_label_save_path),mask_4)



            # 设置Loss检测，当检测到pixel_accuracy停止下降时,调整loss
            if cfg.SOLVER.LR_SCHEDULER == "StepLR":
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                scheduler.step()
                new_lr = optimizer.state_dict()['param_groups'][0]['lr']
                    
            # elif cfg.SOLVER.LR_SCHEDULER == "ReduceLROnPlateau":
            #     lr = optimizer.state_dict()['param_groups'][0]['lr']
            #     scheduler.step(-avg_loss)
            #     new_lr = optimizer.state_dict()['param_groups'][0]['lr']
            #     print(new_lr,lr)
            #     if new_lr != lr:
            #         cfg.SOLVER.LR_SCHEDULER_REPEAT = cfg.SOLVER.LR_SCHEDULER_REPEAT - 1
            #         if cfg.SOLVER.LR_SCHEDULER_REPEAT <0:  trainer.terminate()   #设定学习率调整次数，降低太多次学习率太低时，终止训练

            elif  cfg.SOLVER.LR_SCHEDULER == "CosineAnnealingLR":
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                scheduler.step()
                new_lr = optimizer.state_dict()['param_groups'][0]['lr']
                pass
            if new_lr!=lr:
                print(new_lr,lr)



    # @trainer.on(Events.EPOCH_COMPLETED)
    # def log_training_result(engine):
    #     if engine.state.epoch % 5 == 0:
    #         evaluator_trainer.run(train_loader)
    #         metrics = evaluator_trainer.state.metrics
    #         avg_loss = metrics["pixel_error"]
    #         logger.info("Training Result - Epoch: {} Avg Pixel Error: {:.7f} ".format(engine.state.epoch,avg_loss))


            
    


    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info("Epoch {} done.Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(engine.state.epoch,timer.value()*timer.step_count,
    		train_loader.batch_size / timer.value()))
        timer.reset()


    # def score_pixel_error(engine):
    # 	error = evaluator.state.metrics['pixel_error']
    # 	return error
    # handler_ModelCheckpoint_pixel_error = ModelCheckpoint(dirname=cfg.OUTPUT.DIR_NAME+"model/",filename_prefix=cfg.TAG+"_"+cfg.MODEL.NET_NAME,
    # 	score_function=score_pixel_error,n_saved=cfg.OUTPUT.N_SAVED,create_dir=True,score_name=cfg.SOLVER.CRITERION,require_empty=False)
    # evaluator.add_event_handler(Events.EPOCH_COMPLETED,handler_ModelCheckpoint_pixel_error,{'model':model.module.state_dict()})
    

    trainer.run(train_loader,max_epochs=epochs)

