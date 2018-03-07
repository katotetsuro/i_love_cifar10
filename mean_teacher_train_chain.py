import chainer
import chainer.functions as F
from chainer import reporter
import copy
import transformer

class MeanTeacherTrainChain(chainer.Link):
    def __init__(self, model):
        super().__init__()
        with self.init_scope():
            self.teacher = model
            self.student = model.copy()
            self.student.copyparams(model)

    def __call__(self, xt, xs, t):

        # teacherのpredictionをラベル的に扱う
        with chainer.using_config('train', False):
            yt = self.teacher(xt)
            yt = F.softmax(yt).array

        # 上のwith抜けたらちゃんとstateがpopされることを期待している
        assert chainer.config.train == True

        ys = self.student(xs)

        class_loss = F.softmax_cross_entropy(ys, t)
        yt = F.softmax(yt).array
        ys = F.softmax(ys)
        print(yt, ys)
        consistency_loss = F.mean_squared_error(yt, ys)
        total_loss = class_loss + consistency_loss

        print(class_loss, consistency_loss, total_loss)

        reporter.report({
            'class_loss': class_loss,
            'consistency_loss': consistency_loss,
            'loss': total_loss
         }, self)

        return total_loss

    # trainerの毎ループ呼ばれるExtensionとして使う
    def on_update_finished(self, trainer):
        print('updating teacher model...')
        alpha = min(1 - 1 / (trainer.updater.iteration + 1), 0.97)
        for t, s in zip(self.teacher.parameters(), self.student.parameters()):
            t = t * alpha + s * (1-alpha)
