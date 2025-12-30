(set-logic ALL)

(declare-datatypes ((List 1))
    ((par (T) ((cons (head T) (tail (List T))) (nil)))))


(define-fun-rec list.length.int ((l (List Int))) Int
  (ite (= l (as nil (List Int)))
       0
       (+ 1 (list.length.int (tail l)))))


(define-fun-rec list.get.int ((l (List Int)) (idx Int)) Int
  (ite (< idx 0)
       (list.get.int l (+ (list.length.int l) idx))
  (ite (= idx 0)
       (head l)
       (list.get.int (tail l) (- idx 1)))))


(define-fun-rec list.count.int ((l (List Int)) (val Int)) Int
  (ite (= l (as nil (List Int)))
       0
       (+ (ite (= (head l) val) 1 0)
          (list.count.int (tail l) val))))

(declare-const x (List Int))
(assert (> (list.length.int x) 3))
(assert (and (= (list.length.int x) 5) (= (list.count.int x (list.get.int x 3)) 2)))
(check-sat)
(get-model)