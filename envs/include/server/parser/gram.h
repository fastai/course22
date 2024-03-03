/* A Bison parser, made by GNU Bison 3.7.5.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2021 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

#ifndef YY_BASE_YY_GRAM_H_INCLUDED
# define YY_BASE_YY_GRAM_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int base_yydebug;
#endif

/* Token kinds.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    YYEMPTY = -2,
    YYEOF = 0,                     /* "end of file"  */
    YYerror = 256,                 /* error  */
    YYUNDEF = 257,                 /* "invalid token"  */
    IDENT = 258,                   /* IDENT  */
    FCONST = 259,                  /* FCONST  */
    SCONST = 260,                  /* SCONST  */
    BCONST = 261,                  /* BCONST  */
    XCONST = 262,                  /* XCONST  */
    Op = 263,                      /* Op  */
    ICONST = 264,                  /* ICONST  */
    PARAM = 265,                   /* PARAM  */
    TYPECAST = 266,                /* TYPECAST  */
    DOT_DOT = 267,                 /* DOT_DOT  */
    COLON_EQUALS = 268,            /* COLON_EQUALS  */
    EQUALS_GREATER = 269,          /* EQUALS_GREATER  */
    LESS_EQUALS = 270,             /* LESS_EQUALS  */
    GREATER_EQUALS = 271,          /* GREATER_EQUALS  */
    NOT_EQUALS = 272,              /* NOT_EQUALS  */
    ABORT_P = 273,                 /* ABORT_P  */
    ABSOLUTE_P = 274,              /* ABSOLUTE_P  */
    ACCESS = 275,                  /* ACCESS  */
    ACTION = 276,                  /* ACTION  */
    ADD_P = 277,                   /* ADD_P  */
    ADMIN = 278,                   /* ADMIN  */
    AFTER = 279,                   /* AFTER  */
    AGGREGATE = 280,               /* AGGREGATE  */
    ALL = 281,                     /* ALL  */
    ALSO = 282,                    /* ALSO  */
    ALTER = 283,                   /* ALTER  */
    ALWAYS = 284,                  /* ALWAYS  */
    ANALYSE = 285,                 /* ANALYSE  */
    ANALYZE = 286,                 /* ANALYZE  */
    AND = 287,                     /* AND  */
    ANY = 288,                     /* ANY  */
    ARRAY = 289,                   /* ARRAY  */
    AS = 290,                      /* AS  */
    ASC = 291,                     /* ASC  */
    ASSERTION = 292,               /* ASSERTION  */
    ASSIGNMENT = 293,              /* ASSIGNMENT  */
    ASYMMETRIC = 294,              /* ASYMMETRIC  */
    AT = 295,                      /* AT  */
    ATTACH = 296,                  /* ATTACH  */
    ATTRIBUTE = 297,               /* ATTRIBUTE  */
    AUTHORIZATION = 298,           /* AUTHORIZATION  */
    BACKWARD = 299,                /* BACKWARD  */
    BEFORE = 300,                  /* BEFORE  */
    BEGIN_P = 301,                 /* BEGIN_P  */
    BETWEEN = 302,                 /* BETWEEN  */
    BIGINT = 303,                  /* BIGINT  */
    BINARY = 304,                  /* BINARY  */
    BIT = 305,                     /* BIT  */
    BOOLEAN_P = 306,               /* BOOLEAN_P  */
    BOTH = 307,                    /* BOTH  */
    BY = 308,                      /* BY  */
    CACHE = 309,                   /* CACHE  */
    CALL = 310,                    /* CALL  */
    CALLED = 311,                  /* CALLED  */
    CASCADE = 312,                 /* CASCADE  */
    CASCADED = 313,                /* CASCADED  */
    CASE = 314,                    /* CASE  */
    CAST = 315,                    /* CAST  */
    CATALOG_P = 316,               /* CATALOG_P  */
    CHAIN = 317,                   /* CHAIN  */
    CHAR_P = 318,                  /* CHAR_P  */
    CHARACTER = 319,               /* CHARACTER  */
    CHARACTERISTICS = 320,         /* CHARACTERISTICS  */
    CHECK = 321,                   /* CHECK  */
    CHECKPOINT = 322,              /* CHECKPOINT  */
    CLASS = 323,                   /* CLASS  */
    CLOSE = 324,                   /* CLOSE  */
    CLUSTER = 325,                 /* CLUSTER  */
    COALESCE = 326,                /* COALESCE  */
    COLLATE = 327,                 /* COLLATE  */
    COLLATION = 328,               /* COLLATION  */
    COLUMN = 329,                  /* COLUMN  */
    COLUMNS = 330,                 /* COLUMNS  */
    COMMENT = 331,                 /* COMMENT  */
    COMMENTS = 332,                /* COMMENTS  */
    COMMIT = 333,                  /* COMMIT  */
    COMMITTED = 334,               /* COMMITTED  */
    CONCURRENTLY = 335,            /* CONCURRENTLY  */
    CONFIGURATION = 336,           /* CONFIGURATION  */
    CONFLICT = 337,                /* CONFLICT  */
    CONNECTION = 338,              /* CONNECTION  */
    CONSTRAINT = 339,              /* CONSTRAINT  */
    CONSTRAINTS = 340,             /* CONSTRAINTS  */
    CONTENT_P = 341,               /* CONTENT_P  */
    CONTINUE_P = 342,              /* CONTINUE_P  */
    CONVERSION_P = 343,            /* CONVERSION_P  */
    COPY = 344,                    /* COPY  */
    COST = 345,                    /* COST  */
    CREATE = 346,                  /* CREATE  */
    CROSS = 347,                   /* CROSS  */
    CSV = 348,                     /* CSV  */
    CUBE = 349,                    /* CUBE  */
    CURRENT_P = 350,               /* CURRENT_P  */
    CURRENT_CATALOG = 351,         /* CURRENT_CATALOG  */
    CURRENT_DATE = 352,            /* CURRENT_DATE  */
    CURRENT_ROLE = 353,            /* CURRENT_ROLE  */
    CURRENT_SCHEMA = 354,          /* CURRENT_SCHEMA  */
    CURRENT_TIME = 355,            /* CURRENT_TIME  */
    CURRENT_TIMESTAMP = 356,       /* CURRENT_TIMESTAMP  */
    CURRENT_USER = 357,            /* CURRENT_USER  */
    CURSOR = 358,                  /* CURSOR  */
    CYCLE = 359,                   /* CYCLE  */
    DATA_P = 360,                  /* DATA_P  */
    DATABASE = 361,                /* DATABASE  */
    DAY_P = 362,                   /* DAY_P  */
    DEALLOCATE = 363,              /* DEALLOCATE  */
    DEC = 364,                     /* DEC  */
    DECIMAL_P = 365,               /* DECIMAL_P  */
    DECLARE = 366,                 /* DECLARE  */
    DEFAULT = 367,                 /* DEFAULT  */
    DEFAULTS = 368,                /* DEFAULTS  */
    DEFERRABLE = 369,              /* DEFERRABLE  */
    DEFERRED = 370,                /* DEFERRED  */
    DEFINER = 371,                 /* DEFINER  */
    DELETE_P = 372,                /* DELETE_P  */
    DELIMITER = 373,               /* DELIMITER  */
    DELIMITERS = 374,              /* DELIMITERS  */
    DEPENDS = 375,                 /* DEPENDS  */
    DESC = 376,                    /* DESC  */
    DETACH = 377,                  /* DETACH  */
    DICTIONARY = 378,              /* DICTIONARY  */
    DISABLE_P = 379,               /* DISABLE_P  */
    DISCARD = 380,                 /* DISCARD  */
    DISTINCT = 381,                /* DISTINCT  */
    DO = 382,                      /* DO  */
    DOCUMENT_P = 383,              /* DOCUMENT_P  */
    DOMAIN_P = 384,                /* DOMAIN_P  */
    DOUBLE_P = 385,                /* DOUBLE_P  */
    DROP = 386,                    /* DROP  */
    EACH = 387,                    /* EACH  */
    ELSE = 388,                    /* ELSE  */
    ENABLE_P = 389,                /* ENABLE_P  */
    ENCODING = 390,                /* ENCODING  */
    ENCRYPTED = 391,               /* ENCRYPTED  */
    END_P = 392,                   /* END_P  */
    ENUM_P = 393,                  /* ENUM_P  */
    ESCAPE = 394,                  /* ESCAPE  */
    EVENT = 395,                   /* EVENT  */
    EXCEPT = 396,                  /* EXCEPT  */
    EXCLUDE = 397,                 /* EXCLUDE  */
    EXCLUDING = 398,               /* EXCLUDING  */
    EXCLUSIVE = 399,               /* EXCLUSIVE  */
    EXECUTE = 400,                 /* EXECUTE  */
    EXISTS = 401,                  /* EXISTS  */
    EXPLAIN = 402,                 /* EXPLAIN  */
    EXTENSION = 403,               /* EXTENSION  */
    EXTERNAL = 404,                /* EXTERNAL  */
    EXTRACT = 405,                 /* EXTRACT  */
    FALSE_P = 406,                 /* FALSE_P  */
    FAMILY = 407,                  /* FAMILY  */
    FETCH = 408,                   /* FETCH  */
    FILTER = 409,                  /* FILTER  */
    FIRST_P = 410,                 /* FIRST_P  */
    FLOAT_P = 411,                 /* FLOAT_P  */
    FOLLOWING = 412,               /* FOLLOWING  */
    FOR = 413,                     /* FOR  */
    FORCE = 414,                   /* FORCE  */
    FOREIGN = 415,                 /* FOREIGN  */
    FORWARD = 416,                 /* FORWARD  */
    FREEZE = 417,                  /* FREEZE  */
    FROM = 418,                    /* FROM  */
    FULL = 419,                    /* FULL  */
    FUNCTION = 420,                /* FUNCTION  */
    FUNCTIONS = 421,               /* FUNCTIONS  */
    GENERATED = 422,               /* GENERATED  */
    GLOBAL = 423,                  /* GLOBAL  */
    GRANT = 424,                   /* GRANT  */
    GRANTED = 425,                 /* GRANTED  */
    GREATEST = 426,                /* GREATEST  */
    GROUP_P = 427,                 /* GROUP_P  */
    GROUPING = 428,                /* GROUPING  */
    GROUPS = 429,                  /* GROUPS  */
    HANDLER = 430,                 /* HANDLER  */
    HAVING = 431,                  /* HAVING  */
    HEADER_P = 432,                /* HEADER_P  */
    HOLD = 433,                    /* HOLD  */
    HOUR_P = 434,                  /* HOUR_P  */
    IDENTITY_P = 435,              /* IDENTITY_P  */
    IF_P = 436,                    /* IF_P  */
    ILIKE = 437,                   /* ILIKE  */
    IMMEDIATE = 438,               /* IMMEDIATE  */
    IMMUTABLE = 439,               /* IMMUTABLE  */
    IMPLICIT_P = 440,              /* IMPLICIT_P  */
    IMPORT_P = 441,                /* IMPORT_P  */
    IN_P = 442,                    /* IN_P  */
    INCLUDE = 443,                 /* INCLUDE  */
    INCLUDING = 444,               /* INCLUDING  */
    INCREMENT = 445,               /* INCREMENT  */
    INDEX = 446,                   /* INDEX  */
    INDEXES = 447,                 /* INDEXES  */
    INHERIT = 448,                 /* INHERIT  */
    INHERITS = 449,                /* INHERITS  */
    INITIALLY = 450,               /* INITIALLY  */
    INLINE_P = 451,                /* INLINE_P  */
    INNER_P = 452,                 /* INNER_P  */
    INOUT = 453,                   /* INOUT  */
    INPUT_P = 454,                 /* INPUT_P  */
    INSENSITIVE = 455,             /* INSENSITIVE  */
    INSERT = 456,                  /* INSERT  */
    INSTEAD = 457,                 /* INSTEAD  */
    INT_P = 458,                   /* INT_P  */
    INTEGER = 459,                 /* INTEGER  */
    INTERSECT = 460,               /* INTERSECT  */
    INTERVAL = 461,                /* INTERVAL  */
    INTO = 462,                    /* INTO  */
    INVOKER = 463,                 /* INVOKER  */
    IS = 464,                      /* IS  */
    ISNULL = 465,                  /* ISNULL  */
    ISOLATION = 466,               /* ISOLATION  */
    JOIN = 467,                    /* JOIN  */
    KEY = 468,                     /* KEY  */
    LABEL = 469,                   /* LABEL  */
    LANGUAGE = 470,                /* LANGUAGE  */
    LARGE_P = 471,                 /* LARGE_P  */
    LAST_P = 472,                  /* LAST_P  */
    LATERAL_P = 473,               /* LATERAL_P  */
    LEADING = 474,                 /* LEADING  */
    LEAKPROOF = 475,               /* LEAKPROOF  */
    LEAST = 476,                   /* LEAST  */
    LEFT = 477,                    /* LEFT  */
    LEVEL = 478,                   /* LEVEL  */
    LIKE = 479,                    /* LIKE  */
    LIMIT = 480,                   /* LIMIT  */
    LISTEN = 481,                  /* LISTEN  */
    LOAD = 482,                    /* LOAD  */
    LOCAL = 483,                   /* LOCAL  */
    LOCALTIME = 484,               /* LOCALTIME  */
    LOCALTIMESTAMP = 485,          /* LOCALTIMESTAMP  */
    LOCATION = 486,                /* LOCATION  */
    LOCK_P = 487,                  /* LOCK_P  */
    LOCKED = 488,                  /* LOCKED  */
    LOGGED = 489,                  /* LOGGED  */
    MAPPING = 490,                 /* MAPPING  */
    MATCH = 491,                   /* MATCH  */
    MATERIALIZED = 492,            /* MATERIALIZED  */
    MAXVALUE = 493,                /* MAXVALUE  */
    METHOD = 494,                  /* METHOD  */
    MINUTE_P = 495,                /* MINUTE_P  */
    MINVALUE = 496,                /* MINVALUE  */
    MODE = 497,                    /* MODE  */
    MONTH_P = 498,                 /* MONTH_P  */
    MOVE = 499,                    /* MOVE  */
    NAME_P = 500,                  /* NAME_P  */
    NAMES = 501,                   /* NAMES  */
    NATIONAL = 502,                /* NATIONAL  */
    NATURAL = 503,                 /* NATURAL  */
    NCHAR = 504,                   /* NCHAR  */
    NEW = 505,                     /* NEW  */
    NEXT = 506,                    /* NEXT  */
    NO = 507,                      /* NO  */
    NONE = 508,                    /* NONE  */
    NOT = 509,                     /* NOT  */
    NOTHING = 510,                 /* NOTHING  */
    NOTIFY = 511,                  /* NOTIFY  */
    NOTNULL = 512,                 /* NOTNULL  */
    NOWAIT = 513,                  /* NOWAIT  */
    NULL_P = 514,                  /* NULL_P  */
    NULLIF = 515,                  /* NULLIF  */
    NULLS_P = 516,                 /* NULLS_P  */
    NUMERIC = 517,                 /* NUMERIC  */
    OBJECT_P = 518,                /* OBJECT_P  */
    OF = 519,                      /* OF  */
    OFF = 520,                     /* OFF  */
    OFFSET = 521,                  /* OFFSET  */
    OIDS = 522,                    /* OIDS  */
    OLD = 523,                     /* OLD  */
    ON = 524,                      /* ON  */
    ONLY = 525,                    /* ONLY  */
    OPERATOR = 526,                /* OPERATOR  */
    OPTION = 527,                  /* OPTION  */
    OPTIONS = 528,                 /* OPTIONS  */
    OR = 529,                      /* OR  */
    ORDER = 530,                   /* ORDER  */
    ORDINALITY = 531,              /* ORDINALITY  */
    OTHERS = 532,                  /* OTHERS  */
    OUT_P = 533,                   /* OUT_P  */
    OUTER_P = 534,                 /* OUTER_P  */
    OVER = 535,                    /* OVER  */
    OVERLAPS = 536,                /* OVERLAPS  */
    OVERLAY = 537,                 /* OVERLAY  */
    OVERRIDING = 538,              /* OVERRIDING  */
    OWNED = 539,                   /* OWNED  */
    OWNER = 540,                   /* OWNER  */
    PARALLEL = 541,                /* PARALLEL  */
    PARSER = 542,                  /* PARSER  */
    PARTIAL = 543,                 /* PARTIAL  */
    PARTITION = 544,               /* PARTITION  */
    PASSING = 545,                 /* PASSING  */
    PASSWORD = 546,                /* PASSWORD  */
    PLACING = 547,                 /* PLACING  */
    PLANS = 548,                   /* PLANS  */
    POLICY = 549,                  /* POLICY  */
    POSITION = 550,                /* POSITION  */
    PRECEDING = 551,               /* PRECEDING  */
    PRECISION = 552,               /* PRECISION  */
    PRESERVE = 553,                /* PRESERVE  */
    PREPARE = 554,                 /* PREPARE  */
    PREPARED = 555,                /* PREPARED  */
    PRIMARY = 556,                 /* PRIMARY  */
    PRIOR = 557,                   /* PRIOR  */
    PRIVILEGES = 558,              /* PRIVILEGES  */
    PROCEDURAL = 559,              /* PROCEDURAL  */
    PROCEDURE = 560,               /* PROCEDURE  */
    PROCEDURES = 561,              /* PROCEDURES  */
    PROGRAM = 562,                 /* PROGRAM  */
    PUBLICATION = 563,             /* PUBLICATION  */
    QUOTE = 564,                   /* QUOTE  */
    RANGE = 565,                   /* RANGE  */
    READ = 566,                    /* READ  */
    REAL = 567,                    /* REAL  */
    REASSIGN = 568,                /* REASSIGN  */
    RECHECK = 569,                 /* RECHECK  */
    RECURSIVE = 570,               /* RECURSIVE  */
    REF_P = 571,                   /* REF_P  */
    REFERENCES = 572,              /* REFERENCES  */
    REFERENCING = 573,             /* REFERENCING  */
    REFRESH = 574,                 /* REFRESH  */
    REINDEX = 575,                 /* REINDEX  */
    RELATIVE_P = 576,              /* RELATIVE_P  */
    RELEASE = 577,                 /* RELEASE  */
    RENAME = 578,                  /* RENAME  */
    REPEATABLE = 579,              /* REPEATABLE  */
    REPLACE = 580,                 /* REPLACE  */
    REPLICA = 581,                 /* REPLICA  */
    RESET = 582,                   /* RESET  */
    RESTART = 583,                 /* RESTART  */
    RESTRICT = 584,                /* RESTRICT  */
    RETURNING = 585,               /* RETURNING  */
    RETURNS = 586,                 /* RETURNS  */
    REVOKE = 587,                  /* REVOKE  */
    RIGHT = 588,                   /* RIGHT  */
    ROLE = 589,                    /* ROLE  */
    ROLLBACK = 590,                /* ROLLBACK  */
    ROLLUP = 591,                  /* ROLLUP  */
    ROUTINE = 592,                 /* ROUTINE  */
    ROUTINES = 593,                /* ROUTINES  */
    ROW = 594,                     /* ROW  */
    ROWS = 595,                    /* ROWS  */
    RULE = 596,                    /* RULE  */
    SAVEPOINT = 597,               /* SAVEPOINT  */
    SCHEMA = 598,                  /* SCHEMA  */
    SCHEMAS = 599,                 /* SCHEMAS  */
    SCROLL = 600,                  /* SCROLL  */
    SEARCH = 601,                  /* SEARCH  */
    SECOND_P = 602,                /* SECOND_P  */
    SECURITY = 603,                /* SECURITY  */
    SELECT = 604,                  /* SELECT  */
    SEQUENCE = 605,                /* SEQUENCE  */
    SEQUENCES = 606,               /* SEQUENCES  */
    SERIALIZABLE = 607,            /* SERIALIZABLE  */
    SERVER = 608,                  /* SERVER  */
    SESSION = 609,                 /* SESSION  */
    SESSION_USER = 610,            /* SESSION_USER  */
    SET = 611,                     /* SET  */
    SETS = 612,                    /* SETS  */
    SETOF = 613,                   /* SETOF  */
    SHARE = 614,                   /* SHARE  */
    SHOW = 615,                    /* SHOW  */
    SIMILAR = 616,                 /* SIMILAR  */
    SIMPLE = 617,                  /* SIMPLE  */
    SKIP = 618,                    /* SKIP  */
    SMALLINT = 619,                /* SMALLINT  */
    SNAPSHOT = 620,                /* SNAPSHOT  */
    SOME = 621,                    /* SOME  */
    SQL_P = 622,                   /* SQL_P  */
    STABLE = 623,                  /* STABLE  */
    STANDALONE_P = 624,            /* STANDALONE_P  */
    START = 625,                   /* START  */
    STATEMENT = 626,               /* STATEMENT  */
    STATISTICS = 627,              /* STATISTICS  */
    STDIN = 628,                   /* STDIN  */
    STDOUT = 629,                  /* STDOUT  */
    STORAGE = 630,                 /* STORAGE  */
    STORED = 631,                  /* STORED  */
    STRICT_P = 632,                /* STRICT_P  */
    STRIP_P = 633,                 /* STRIP_P  */
    SUBSCRIPTION = 634,            /* SUBSCRIPTION  */
    SUBSTRING = 635,               /* SUBSTRING  */
    SUPPORT = 636,                 /* SUPPORT  */
    SYMMETRIC = 637,               /* SYMMETRIC  */
    SYSID = 638,                   /* SYSID  */
    SYSTEM_P = 639,                /* SYSTEM_P  */
    TABLE = 640,                   /* TABLE  */
    TABLES = 641,                  /* TABLES  */
    TABLESAMPLE = 642,             /* TABLESAMPLE  */
    TABLESPACE = 643,              /* TABLESPACE  */
    TEMP = 644,                    /* TEMP  */
    TEMPLATE = 645,                /* TEMPLATE  */
    TEMPORARY = 646,               /* TEMPORARY  */
    TEXT_P = 647,                  /* TEXT_P  */
    THEN = 648,                    /* THEN  */
    TIES = 649,                    /* TIES  */
    TIME = 650,                    /* TIME  */
    TIMESTAMP = 651,               /* TIMESTAMP  */
    TO = 652,                      /* TO  */
    TRAILING = 653,                /* TRAILING  */
    TRANSACTION = 654,             /* TRANSACTION  */
    TRANSFORM = 655,               /* TRANSFORM  */
    TREAT = 656,                   /* TREAT  */
    TRIGGER = 657,                 /* TRIGGER  */
    TRIM = 658,                    /* TRIM  */
    TRUE_P = 659,                  /* TRUE_P  */
    TRUNCATE = 660,                /* TRUNCATE  */
    TRUSTED = 661,                 /* TRUSTED  */
    TYPE_P = 662,                  /* TYPE_P  */
    TYPES_P = 663,                 /* TYPES_P  */
    UNBOUNDED = 664,               /* UNBOUNDED  */
    UNCOMMITTED = 665,             /* UNCOMMITTED  */
    UNENCRYPTED = 666,             /* UNENCRYPTED  */
    UNION = 667,                   /* UNION  */
    UNIQUE = 668,                  /* UNIQUE  */
    UNKNOWN = 669,                 /* UNKNOWN  */
    UNLISTEN = 670,                /* UNLISTEN  */
    UNLOGGED = 671,                /* UNLOGGED  */
    UNTIL = 672,                   /* UNTIL  */
    UPDATE = 673,                  /* UPDATE  */
    USER = 674,                    /* USER  */
    USING = 675,                   /* USING  */
    VACUUM = 676,                  /* VACUUM  */
    VALID = 677,                   /* VALID  */
    VALIDATE = 678,                /* VALIDATE  */
    VALIDATOR = 679,               /* VALIDATOR  */
    VALUE_P = 680,                 /* VALUE_P  */
    VALUES = 681,                  /* VALUES  */
    VARCHAR = 682,                 /* VARCHAR  */
    VARIADIC = 683,                /* VARIADIC  */
    VARYING = 684,                 /* VARYING  */
    VERBOSE = 685,                 /* VERBOSE  */
    VERSION_P = 686,               /* VERSION_P  */
    VIEW = 687,                    /* VIEW  */
    VIEWS = 688,                   /* VIEWS  */
    VOLATILE = 689,                /* VOLATILE  */
    WHEN = 690,                    /* WHEN  */
    WHERE = 691,                   /* WHERE  */
    WHITESPACE_P = 692,            /* WHITESPACE_P  */
    WINDOW = 693,                  /* WINDOW  */
    WITH = 694,                    /* WITH  */
    WITHIN = 695,                  /* WITHIN  */
    WITHOUT = 696,                 /* WITHOUT  */
    WORK = 697,                    /* WORK  */
    WRAPPER = 698,                 /* WRAPPER  */
    WRITE = 699,                   /* WRITE  */
    XML_P = 700,                   /* XML_P  */
    XMLATTRIBUTES = 701,           /* XMLATTRIBUTES  */
    XMLCONCAT = 702,               /* XMLCONCAT  */
    XMLELEMENT = 703,              /* XMLELEMENT  */
    XMLEXISTS = 704,               /* XMLEXISTS  */
    XMLFOREST = 705,               /* XMLFOREST  */
    XMLNAMESPACES = 706,           /* XMLNAMESPACES  */
    XMLPARSE = 707,                /* XMLPARSE  */
    XMLPI = 708,                   /* XMLPI  */
    XMLROOT = 709,                 /* XMLROOT  */
    XMLSERIALIZE = 710,            /* XMLSERIALIZE  */
    XMLTABLE = 711,                /* XMLTABLE  */
    YEAR_P = 712,                  /* YEAR_P  */
    YES_P = 713,                   /* YES_P  */
    ZONE = 714,                    /* ZONE  */
    NOT_LA = 715,                  /* NOT_LA  */
    NULLS_LA = 716,                /* NULLS_LA  */
    WITH_LA = 717,                 /* WITH_LA  */
    POSTFIXOP = 718,               /* POSTFIXOP  */
    UMINUS = 719                   /* UMINUS  */
  };
  typedef enum yytokentype yytoken_kind_t;
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#line 203 "gram.y"

	core_YYSTYPE		core_yystype;
	/* these fields must match core_YYSTYPE: */
	int					ival;
	char				*str;
	const char			*keyword;

	char				chr;
	bool				boolean;
	JoinType			jtype;
	DropBehavior		dbehavior;
	OnCommitAction		oncommit;
	List				*list;
	Node				*node;
	Value				*value;
	ObjectType			objtype;
	TypeName			*typnam;
	FunctionParameter   *fun_param;
	FunctionParameterMode fun_param_mode;
	ObjectWithArgs		*objwithargs;
	DefElem				*defelt;
	SortBy				*sortby;
	WindowDef			*windef;
	JoinExpr			*jexpr;
	IndexElem			*ielem;
	Alias				*alias;
	RangeVar			*range;
	IntoClause			*into;
	WithClause			*with;
	InferClause			*infer;
	OnConflictClause	*onconflict;
	A_Indices			*aind;
	ResTarget			*target;
	struct PrivTarget	*privtarget;
	AccessPriv			*accesspriv;
	struct ImportQual	*importqual;
	InsertStmt			*istmt;
	VariableSetStmt		*vsetstmt;
	PartitionElem		*partelem;
	PartitionSpec		*partspec;
	PartitionBoundSpec	*partboundspec;
	RoleSpec			*rolespec;

#line 572 "gram.h"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif

/* Location type.  */
#if ! defined YYLTYPE && ! defined YYLTYPE_IS_DECLARED
typedef struct YYLTYPE YYLTYPE;
struct YYLTYPE
{
  int first_line;
  int first_column;
  int last_line;
  int last_column;
};
# define YYLTYPE_IS_DECLARED 1
# define YYLTYPE_IS_TRIVIAL 1
#endif



int base_yyparse (core_yyscan_t yyscanner);

#endif /* !YY_BASE_YY_GRAM_H_INCLUDED  */
