import chainer
import chainer.functions as F
from chainer import reporter
import copy
import transformer

class MeanTeacherTrainChain(chainer.Chain):
    def __init__(self, teacher, student):
        super().__init__()
        self.teacher = teacher
        with self.init_scope():
            self.student = student

    def __call__(self, xt, xs, t):

        # teacherのpredictionをラベル的に扱う
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            yt = self.teacher(xt)
            yt = F.softmax(yt).array
            acc_t = F.accuracy(yt, t)

        # 上のwith抜けたらちゃんとstateがpopされることを期待している
        assert chainer.config.train == True

        ys = self.student(xs)

        class_loss = F.softmax_cross_entropy(ys, t)
        ys = F.softmax(ys)
        consistency_loss = F.mean_squared_error(yt, ys) * 100
        total_loss = class_loss + consistency_loss
        acc_s = F.accuracy(ys, t)

        reporter.report({
            'class_loss': class_loss,
            'consistency_loss': consistency_loss,
            'loss': total_loss,
            'teacher_accuracy': acc_t,
            'student_accuracy': acc_s
        }, self)

        return total_loss

    def recursive_copy(self, t, s, alpha):
        if isinstance(t, chainer.Chain):
            for c in t._children:
                self.recursive_copy(t[c], s[c], alpha)
        elif isinstance(t, chainer.Link):
            for name in t._params:
                t.__dict__[name].array = t.__dict__[name].array * alpha + s.__dict__[name].array * (1-alpha)

    # trainerの毎ループ呼ばれるExtensionとして使う
    def on_update_finished(self, trainer):
        alpha = min(1 - 1 / (trainer.updater.iteration + 1), 0.97)
        #with chainer.no_backprop_mode():
        self.recursive_copy(self.teacher, self.student, alpha)
