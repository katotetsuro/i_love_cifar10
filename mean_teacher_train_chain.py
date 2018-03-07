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
        consistency_loss = F.mean_squared_error(yt, ys)
        total_loss = class_loss + consistency_loss

        reporter.report({
            'class_loss': class_loss,
            'consistency_loss': consistency_loss,
            'loss': total_loss
         }, self)

        return total_loss

    # trainerの毎ループ呼ばれるExtensionとして使う
    def on_update_finished(self, trainer):
        alpha = min(1 - 1 / (trainer.updater.iteration + 1), 0.97)
        # https://github.com/chainer/chainer/blob/v3.4.0/chainer/link.py#L450
        t = self.teacher.__dict__
        s = self.student.__dict__
        for name in self.teacher._params:
            t[name] = t[name] * alpha + s[name] * (1-alpha)
