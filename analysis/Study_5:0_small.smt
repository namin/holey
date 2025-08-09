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

(define-fun-rec list.contains.int ((l (List Int)) (val Int)) Bool
  (ite (= l (as nil (List Int)))
       false
       (ite (= (head l) val)
            true
            (list.contains.int (tail l) val))))

(define-fun-rec list.index.rec.int ((i Int) (l (List Int)) (val Int)) Int
  (ite (= l (as nil (List Int)))
       -1
       (ite (= (head l) val)
            i
            (list.index.rec.int (+ 1 i) (tail l) val))))

(define-fun list.index.int ((l (List Int)) (val Int)) Int
  (list.index.rec.int 0 l val))

(define-fun-rec list.length.string ((l (List String))) Int
  (ite (= l (as nil (List String)))
       0
       (+ 1 (list.length.string (tail l)))))

(define-fun-rec list.get.string ((l (List String)) (idx Int)) String
  (ite (< idx 0)
       (list.get.string l (+ (list.length.string l) idx))
  (ite (= idx 0)
       (head l)
       (list.get.string (tail l) (- idx 1)))))

(define-fun-rec list.contains.string ((l (List String)) (val String)) Bool
  (ite (= l (as nil (List String)))
       false
       (ite (= (head l) val)
            true
            (list.contains.string (tail l) val))))

(define-fun-rec list.sum.int ((l (List Int))) Int
  (ite (= l (as nil (List Int)))
       0
       (+ (head l) (list.sum.int (tail l)))))

(define-fun-rec list.append.int ((l1 (List Int)) (l2 (List Int))) (List Int)
  (ite (= l1 (as nil (List Int)))
       l2
       (cons (head l1) (list.append.int (tail l1) l2))))

(define-fun-rec list.append.string ((l1 (List String)) (l2 (List String))) (List String)
  (ite (= l1 (as nil (List String)))
       l2
       (cons (head l1) (list.append.string (tail l1) l2))))

(define-fun-rec list.map_add.int ((l (List Int)) (val Int)) (List Int)
  (ite (= l (as nil (List Int)))
       (as nil (List Int))
       (cons (+ (head l) val) (list.map_add.int (tail l) val))))

(define-fun-rec list.count.int ((l (List Int)) (val Int)) Int
  (ite (= l (as nil (List Int)))
       0
       (+ (ite (= (head l) val) 1 0)
          (list.count.int (tail l) val))))

(define-fun-rec list.count.string ((l (List String)) (val String)) Int
  (ite (= l (as nil (List String)))
       0
       (+ (ite (= (head l) val) 1 0)
          (list.count.string (tail l) val))))

(define-fun-rec list.count.bool ((l (List Bool)) (val Bool)) Int
  (ite (= l (as nil (List Bool)))
       0
       (+ (ite (= (head l) val) 1 0)
          (list.count.bool (tail l) val))))

(define-fun-rec list.count.real ((l (List Real)) (val Real)) Int
  (ite (= l (as nil (List Real)))
       0
       (+ (ite (= (head l) val) 1 0)
          (list.count.real (tail l) val))))

(declare-const x (List Int))
(assert (and
         (= (list.count.int x 0) 0)
         (= (list.count.int x 1) 1)
         (= (list.count.int x 2) 2)
         (= (list.count.int x 3) 3)
         ))
(check-sat)
(get-model)
